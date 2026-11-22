"""Strict claim parsing: validate fields and types, normalize codes."""
import math
from datetime import date

from claims_review.models import ParsedClaim


class ClaimParseError(ValueError):
    """Raised when a claim dict is missing a field or has a wrong type or value."""


def _label(raw) -> str:
    cid = raw.get("claim_id") if isinstance(raw, dict) else None
    return cid if isinstance(cid, str) and cid.strip() else "(no claim_id)"


def _text(raw: dict, key: str, label: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ClaimParseError(f"{label}: field '{key}' must be a non-empty string")
    return value.strip()


def _code(value, what: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ClaimParseError(f"{label}: {what} must be a non-empty string")
    return value.strip().upper()


def parse_claim(raw: dict) -> ParsedClaim:
    """Validate one raw claim dict and return a ParsedClaim, or raise ClaimParseError."""
    if not isinstance(raw, dict):
        raise ClaimParseError(f"claim must be a dict, got {type(raw).__name__}")
    label = _label(raw)

    provider = raw.get("provider")
    if not isinstance(provider, dict):
        raise ClaimParseError(f"{label}: field 'provider' must be an object with name and specialty")
    name = _text(provider, "name", f"{label} provider")
    specialty = _text(provider, "specialty", f"{label} provider")

    dos_text = _text(raw, "date_of_service", label)
    try:
        dos = date.fromisoformat(dos_text)
    except ValueError:
        raise ClaimParseError(f"{label}: date_of_service '{dos_text}' is not a valid YYYY-MM-DD date") from None
    if dos.isoformat() != dos_text:
        raise ClaimParseError(f"{label}: date_of_service '{dos_text}' must be written as YYYY-MM-DD")

    modifiers = raw.get("modifiers")
    if not isinstance(modifiers, list):
        raise ClaimParseError(f"{label}: field 'modifiers' must be a list")
    icd10 = raw.get("icd10")
    if not isinstance(icd10, list) or not icd10:
        raise ClaimParseError(f"{label}: field 'icd10' must be a non-empty list")

    units = raw.get("units")
    if not isinstance(units, int) or isinstance(units, bool) or units < 1:
        raise ClaimParseError(f"{label}: field 'units' must be a whole number of at least 1")
    billed = raw.get("billed_amount")
    if (not isinstance(billed, (int, float)) or isinstance(billed, bool)
            or not math.isfinite(billed) or billed <= 0):
        raise ClaimParseError(f"{label}: field 'billed_amount' must be a positive finite number")

    note = raw.get("clinical_note")
    if not isinstance(note, str):
        raise ClaimParseError(f"{label}: field 'clinical_note' must be a string")

    return ParsedClaim(
        claim_id=_text(raw, "claim_id", label),
        patient_ref=_text(raw, "patient_ref", label),
        patient_age_band=_text(raw, "patient_age_band", label),
        patient_sex=_text(raw, "patient_sex", label),
        date_of_service=dos,
        provider_name=name,
        provider_specialty=specialty,
        place_of_service=_text(raw, "place_of_service", label),
        cpt=_code(raw.get("cpt"), "field 'cpt'", label),
        modifiers=tuple(_code(m, "a modifier", label) for m in modifiers),
        icd10=tuple(_code(i, "an icd10 code", label) for i in icd10),
        units=units,
        billed_amount=float(billed),
        clinical_note=note,
    )
