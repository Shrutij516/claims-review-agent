"""Helpers for building tiny hand-made claims in tests."""
from claims_review.parsing import parse_claim
from claims_review.rules import rule_flags


def raw(**over):
    """A valid raw claim dict that raises no flag; keyword overrides replace fields."""
    base = {
        "claim_id": "T-0001",
        "patient_ref": "P-9001",
        "patient_age_band": "40-49",
        "patient_sex": "F",
        "date_of_service": "2026-02-10",
        "provider": {"name": "Dr. Test One", "specialty": "Family Medicine"},
        "place_of_service": "11",
        "cpt": "99213",
        "modifiers": [],
        "icd10": ["J06.9"],
        "units": 1,
        "billed_amount": 100.0,
        "clinical_note": "Routine visit. Nothing unusual.",
    }
    if "provider_name" in over:
        base["provider"] = {"name": over.pop("provider_name"), "specialty": "Family Medicine"}
    base.update(over)
    return base


def make(**over):
    """Parse a raw claim built with raw()."""
    return parse_claim(raw(**over))


def flag_ids(claim, store=None):
    """The set of rule flag ids for a claim; store defaults to just that claim."""
    return {f.flag_id for f in rule_flags(claim, store if store is not None else [claim])}
