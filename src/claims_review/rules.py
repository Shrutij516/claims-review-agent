"""The eight deterministic R_ rule checks. Thresholds come from data/v2/flag_catalog.json."""
import json
from functools import lru_cache

from claims_review import DATA_V2
from claims_review.codes import CodeTables, load_tables, validate_codes
from claims_review.models import Flag, ParsedClaim

RULE_FLAG_IDS = (
    "R_INVALID_CODE",
    "R_DUPLICATE_CLAIM",
    "R_UNITS_EXCEEDED",
    "R_PRICE_OUTLIER",
    "R_LATERALITY_MISSING",
    "R_DX_NOT_SUPPORTING",
    "R_MOD25_MISSING",
    "R_EM_LEVEL_VS_DX",
)


@lru_cache(maxsize=None)
def load_catalog() -> dict:
    """Load the flag catalog (flags plus the synthetic parameters block)."""
    return json.loads((DATA_V2 / "flag_catalog.json").read_text(encoding="utf-8"))


def load_params() -> dict:
    """Return the parameters block of the flag catalog."""
    return load_catalog()["parameters"]


def policy_criteria_for(flag_id: str, cpt: str, tables: CodeTables) -> list[str]:
    """Criterion ids that explain a flag for this procedure code (the catalog only lists policy ids)."""
    if flag_id == "R_DUPLICATE_CLAIM":
        return ["POL-004.1"]
    if flag_id == "R_MOD25_MISSING":
        return ["POL-006.1"]
    if flag_id == "R_EM_LEVEL_VS_DX":
        return ["POL-002.2"]
    if flag_id == "R_LATERALITY_MISSING":
        return ["POL-003.4"] if cpt == "73721" else ["POL-001.2"]
    if flag_id == "R_DX_NOT_SUPPORTING":
        return ["POL-001.1"] if cpt == "29881" else []
    if flag_id == "R_UNITS_EXCEEDED":
        if cpt == "97110":
            return ["POL-005.1", "POL-005.2"]
        if cpt in tables.hcpcs:
            return ["POL-005.1", "POL-007.3"]
        return ["POL-005.1"]
    return []


def _make_flag(flag_id: str, detail: str, cpt: str, tables: CodeTables, extra: list[dict] | None = None) -> Flag:
    evidence = [{"kind": "flag", "id": flag_id}]
    evidence += [{"kind": "policy", "id": c} for c in policy_criteria_for(flag_id, cpt, tables)]
    evidence += extra or []
    return Flag(flag_id=flag_id, detail=detail, evidence=evidence)


def rule_flags(parsed: ParsedClaim, store: list[ParsedClaim],
               tables: CodeTables | None = None, params: dict | None = None) -> list[Flag]:
    """Return the R_ flags for one claim. store holds every claim (needed for duplicate and modifier 25 checks).

    If the procedure code is not in the tables, only R_INVALID_CODE is returned,
    because every other check depends on knowing the procedure.
    """
    tables = tables or load_tables()
    params = params or load_params()
    checks = validate_codes(parsed, tables)
    procedure = checks[0]
    cpt = parsed.cpt
    found: dict[str, Flag] = {}

    bad = [c for c in checks if not c.valid]
    if bad:
        listing = ", ".join(f"{c.code} ({c.system})" for c in bad)
        found["R_INVALID_CODE"] = _make_flag(
            "R_INVALID_CODE", f"Code not in the code tables: {listing}", cpt, tables,
            extra=[{"kind": "code", "id": c.code} for c in bad])
    if not procedure.valid:
        return list(found.values())

    valid_icds = [c for c in checks if c.system == "icd10" and c.valid]
    others = [o for o in store if o.claim_id != parsed.claim_id and o.patient_ref == parsed.patient_ref]

    # Duplicate: same patient, procedure and provider, an earlier claim within the window.
    window = params["duplicate_window_days"]
    me = (parsed.date_of_service, parsed.claim_id)
    for o in others:
        gap = (parsed.date_of_service - o.date_of_service).days
        if (o.cpt == cpt and o.provider_name == parsed.provider_name
                and (o.date_of_service, o.claim_id) < me and gap <= window):
            found["R_DUPLICATE_CLAIM"] = _make_flag(
                "R_DUPLICATE_CLAIM",
                f"Same patient, code and provider as {o.claim_id}, {gap} days earlier (window {window})", cpt, tables)
            break

    # Units above the per-code daily limit (exactly at the limit is allowed).
    limit = params["units_per_day_limit"].get(cpt)
    if limit is not None and parsed.units > limit:
        found["R_UNITS_EXCEEDED"] = _make_flag(
            "R_UNITS_EXCEEDED", f"{parsed.units} units billed, daily limit is {limit}", cpt, tables)

    # Price per unit above the multiplier times the top of the synthetic band.
    band = params["price_bands_per_unit"].get(cpt)
    if band is not None:
        per_unit = parsed.billed_amount / parsed.units
        ceiling = params["price_outlier_multiplier"] * band[1]
        if per_unit > ceiling:
            found["R_PRICE_OUTLIER"] = _make_flag(
                "R_PRICE_OUTLIER",
                f"{per_unit:.2f} per unit is above the ceiling {ceiling:.2f} (band top {band[1]})", cpt, tables)

    # Laterality: needs an RT or LT modifier, or a diagnosis that names a side.
    if cpt in params["laterality_required_codes"]:
        by_modifier = any(m in ("RT", "LT") for m in parsed.modifiers)
        by_diagnosis = any(c.laterality in ("right", "left", "bilateral") for c in valid_icds)
        if not (by_modifier or by_diagnosis):
            found["R_LATERALITY_MISSING"] = _make_flag(
                "R_LATERALITY_MISSING", "No RT or LT modifier and no diagnosis code that gives the side", cpt, tables)

    # Diagnosis support list for the procedure.
    prefixes = params["dx_support_prefixes"].get(cpt)
    if prefixes and valid_icds and not any(c.code.startswith(p) for c in valid_icds for p in prefixes):
        found["R_DX_NOT_SUPPORTING"] = _make_flag(
            "R_DX_NOT_SUPPORTING",
            f"No diagnosis starts with any of {', '.join(prefixes)}", cpt, tables)

    # Modifier 25 on an office visit billed with a same-day procedure.
    if cpt in params["mod25_em_codes"] and "25" not in parsed.modifiers:
        for o in others:
            if (o.provider_name == parsed.provider_name and o.date_of_service == parsed.date_of_service
                    and o.cpt in params["mod25_procedure_codes"]):
                found["R_MOD25_MISSING"] = _make_flag(
                    "R_MOD25_MISSING", f"Same-day procedure {o.cpt} in {o.claim_id} but no modifier 25", cpt, tables)
                break

    # Office visit level needs a diagnosis of at least the required complexity tier.
    needed = params["em_min_tier"].get(cpt)
    if needed is not None:
        tiers = [params["icd_tier"][c.code] for c in valid_icds if c.code in params["icd_tier"]]
        if tiers and max(tiers) < needed:
            found["R_EM_LEVEL_VS_DX"] = _make_flag(
                "R_EM_LEVEL_VS_DX", f"Level needs tier {needed}, highest diagnosis tier is {max(tiers)}", cpt, tables)

    return [found[f] for f in RULE_FLAG_IDS if f in found]
