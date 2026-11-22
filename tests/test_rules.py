import json
from pathlib import Path

import pytest

from claims_review.codes import load_tables
from claims_review.parsing import parse_claim
from claims_review.retrieval import get_criterion
from claims_review.rules import RULE_FLAG_IDS, load_catalog, rule_flags
from factories import flag_ids, make

ROOT = Path(__file__).resolve().parent.parent


def test_default_claim_raises_no_flag():
    assert flag_ids(make()) == set()


# ---------------------------------------------------------------- R_INVALID_CODE
def test_invalid_procedure_code_flags_only_invalid_code():
    claim = make(cpt="99216", units=50, billed_amount=99999.0)
    flags = rule_flags(claim, [claim])
    assert [f.flag_id for f in flags] == ["R_INVALID_CODE"]
    assert {"kind": "code", "id": "99216"} in flags[0].evidence


def test_invalid_diagnosis_and_modifier_flag_but_other_rules_still_run():
    assert "R_INVALID_CODE" in flag_ids(make(icd10=["I10", "I1O"]))
    assert "R_INVALID_CODE" in flag_ids(make(modifiers=["ZZ"]))
    both = flag_ids(make(cpt="99215", icd10=["J06.9", "I1O"]))
    assert {"R_INVALID_CODE", "R_EM_LEVEL_VS_DX"} <= both


# ------------------------------------------------------------- R_DUPLICATE_CLAIM
def _pair(day2, **second):
    a = make(claim_id="A-1", date_of_service="2026-02-01")
    b = make(claim_id="A-2", date_of_service=day2, **second)
    return a, b, [a, b]


def test_duplicate_at_14_days_is_flagged_and_at_15_is_not():
    a, b, store = _pair("2026-02-15")
    assert "R_DUPLICATE_CLAIM" in flag_ids(b, store)
    a, b, store = _pair("2026-02-16")
    assert "R_DUPLICATE_CLAIM" not in flag_ids(b, store)


def test_only_the_later_claim_of_a_pair_is_flagged():
    a, b, store = _pair("2026-02-05")
    assert "R_DUPLICATE_CLAIM" not in flag_ids(a, store)
    assert "R_DUPLICATE_CLAIM" in flag_ids(b, store)


def test_same_day_pair_flags_the_higher_claim_id_only():
    a, b, store = _pair("2026-02-01")
    assert "R_DUPLICATE_CLAIM" not in flag_ids(a, store)
    assert "R_DUPLICATE_CLAIM" in flag_ids(b, store)


@pytest.mark.parametrize("override", [
    {"provider_name": "Dr. Other"},
    {"patient_ref": "P-9002"},
    {"cpt": "99214", "icd10": ["I10"], "billed_amount": 150.0},
])
def test_not_a_duplicate_when_provider_patient_or_code_differs(override):
    a, b, store = _pair("2026-02-05", **override)
    assert "R_DUPLICATE_CLAIM" not in flag_ids(b, store)


# ---------------------------------------------------------------- R_UNITS_EXCEEDED
def test_units_at_limit_allowed_and_above_limit_flagged_for_97110():
    assert "R_UNITS_EXCEEDED" not in flag_ids(make(cpt="97110", icd10=["M54.50"], units=4, billed_amount=200.0))
    assert "R_UNITS_EXCEEDED" in flag_ids(make(cpt="97110", icd10=["M54.50"], units=5, billed_amount=250.0))


def test_units_limit_for_a_drug_code():
    assert "R_UNITS_EXCEEDED" not in flag_ids(make(cpt="J3301", icd10=["M25.511"], units=8, billed_amount=80.0))
    assert "R_UNITS_EXCEEDED" in flag_ids(make(cpt="J3301", icd10=["M25.511"], units=9, billed_amount=90.0))


def test_units_limit_for_an_office_visit_is_one():
    assert "R_UNITS_EXCEEDED" in flag_ids(make(units=2, billed_amount=200.0))


# ----------------------------------------------------------------- R_PRICE_OUTLIER
def test_price_exactly_at_the_ceiling_is_allowed_and_a_cent_above_is_flagged():
    # 99213 band top is 140, multiplier 1.5, ceiling 210.00
    assert "R_PRICE_OUTLIER" not in flag_ids(make(billed_amount=210.00))
    assert "R_PRICE_OUTLIER" in flag_ids(make(billed_amount=210.01))


def test_price_is_judged_per_unit():
    # 97110 band top is 60, ceiling 90 per unit
    assert "R_PRICE_OUTLIER" not in flag_ids(make(cpt="97110", icd10=["M54.50"], units=4, billed_amount=360.0))
    assert "R_PRICE_OUTLIER" in flag_ids(make(cpt="97110", icd10=["M54.50"], units=4, billed_amount=364.0))


def test_low_price_is_not_flagged():
    assert "R_PRICE_OUTLIER" not in flag_ids(make(billed_amount=1.0))


# ------------------------------------------------------------ R_LATERALITY_MISSING
def test_29881_needs_a_side_from_modifier_or_diagnosis():
    base = {"cpt": "29881", "billed_amount": 2500.0}
    assert "R_LATERALITY_MISSING" in flag_ids(make(icd10=["M23.229"], **base))
    assert "R_LATERALITY_MISSING" not in flag_ids(make(icd10=["M23.229"], modifiers=["RT"], **base))
    assert "R_LATERALITY_MISSING" not in flag_ids(make(icd10=["M23.229"], modifiers=["LT"], **base))
    assert "R_LATERALITY_MISSING" not in flag_ids(make(icd10=["M23.221"], **base))
    assert "R_LATERALITY_MISSING" not in flag_ids(make(icd10=["M23.222"], **base))


def test_73721_needs_a_side_and_bilateral_counts():
    base = {"cpt": "73721", "billed_amount": 500.0}
    assert "R_LATERALITY_MISSING" in flag_ids(make(icd10=["M25.569"], **base))
    assert "R_LATERALITY_MISSING" not in flag_ids(make(icd10=["M25.561"], **base))
    assert "R_LATERALITY_MISSING" not in flag_ids(make(icd10=["M17.0"], **base))


def test_laterality_not_required_for_other_codes():
    assert "R_LATERALITY_MISSING" not in flag_ids(make(cpt="99213", icd10=["M25.569"]))


# ------------------------------------------------------------ R_DX_NOT_SUPPORTING
def test_diagnosis_support_list_is_checked_per_procedure():
    assert "R_DX_NOT_SUPPORTING" in flag_ids(make(cpt="71046", icd10=["M54.50"], billed_amount=80.0))
    assert "R_DX_NOT_SUPPORTING" not in flag_ids(make(cpt="71046", icd10=["J06.9"], billed_amount=80.0))
    assert "R_DX_NOT_SUPPORTING" not in flag_ids(make(cpt="71046", icd10=["M54.50", "J06.9"], billed_amount=80.0))
    assert "R_DX_NOT_SUPPORTING" in flag_ids(make(cpt="93000", icd10=["J06.9"], billed_amount=40.0))


def test_one_supporting_diagnosis_among_several_is_enough_for_29881():
    claim = make(cpt="29881", icd10=["M17.11", "M23.251"], billed_amount=2500.0)
    assert "R_DX_NOT_SUPPORTING" not in flag_ids(claim)
    assert "R_DX_NOT_SUPPORTING" in flag_ids(make(cpt="29881", icd10=["M17.11"], billed_amount=2500.0))


def test_procedure_without_a_support_list_never_raises_it():
    assert "R_DX_NOT_SUPPORTING" not in flag_ids(make(cpt="99213", icd10=["R51.9"]))


# --------------------------------------------------------------- R_MOD25_MISSING
def _visit_and_procedure(visit_mods=(), proc_over=None, visit_over=None):
    visit = make(claim_id="V-1", cpt="99213", icd10=["M25.511"], modifiers=list(visit_mods), **(visit_over or {}))
    proc = make(claim_id="V-2", cpt="20610", icd10=["M25.511"], billed_amount=100.0, **(proc_over or {}))
    return visit, proc, [visit, proc]


def test_visit_with_same_day_procedure_needs_modifier_25():
    visit, proc, store = _visit_and_procedure()
    assert "R_MOD25_MISSING" in flag_ids(visit, store)
    visit, proc, store = _visit_and_procedure(visit_mods=["25"])
    assert "R_MOD25_MISSING" not in flag_ids(visit, store)


def test_modifier_25_rule_needs_same_day_same_provider_same_patient():
    visit, proc, store = _visit_and_procedure(proc_over={"date_of_service": "2026-02-11"})
    assert "R_MOD25_MISSING" not in flag_ids(visit, store)
    visit, proc, store = _visit_and_procedure(proc_over={"provider_name": "Dr. Other"})
    assert "R_MOD25_MISSING" not in flag_ids(visit, store)
    visit, proc, store = _visit_and_procedure(proc_over={"patient_ref": "P-9002"})
    assert "R_MOD25_MISSING" not in flag_ids(visit, store)


def test_procedure_claim_itself_never_needs_modifier_25():
    visit, proc, store = _visit_and_procedure()
    assert "R_MOD25_MISSING" not in flag_ids(proc, store)


def test_other_same_day_codes_do_not_trigger_modifier_25():
    visit = make(claim_id="V-1", cpt="99213", icd10=["M25.511"])
    other = make(claim_id="V-2", cpt="36415", icd10=["M25.511"], billed_amount=15.0)
    assert "R_MOD25_MISSING" not in flag_ids(visit, [visit, other])


# --------------------------------------------------------------- R_EM_LEVEL_VS_DX
@pytest.mark.parametrize("cpt, icd, billed, flagged", [
    ("99213", ["J06.9"], 100.0, False),   # tier 1 needed, tier 1 present
    ("99214", ["M54.50"], 150.0, True),   # tier 2 needed, only tier 1
    ("99214", ["I10"], 150.0, False),     # tier 2 present
    ("99214", ["M54.50", "I10"], 150.0, False),  # highest tier counts
    ("99215", ["Z00.00"], 250.0, True),   # tier 3 needed, only tier 1
    ("99215", ["I10"], 250.0, True),      # tier 3 needed, only tier 2
    ("99215", ["I50.9"], 250.0, False),   # tier 3 present
    ("99215", ["N18.4"], 250.0, False),
])
def test_em_level_against_diagnosis_tier(cpt, icd, billed, flagged):
    assert ("R_EM_LEVEL_VS_DX" in flag_ids(make(cpt=cpt, icd10=icd, billed_amount=billed))) is flagged


def test_em_rule_does_not_apply_to_non_visit_codes():
    assert "R_EM_LEVEL_VS_DX" not in flag_ids(make(cpt="80053", icd10=["J06.9"], billed_amount=40.0))


# ------------------------------------------------------------- shape and evidence
def test_flag_order_follows_rule_flag_ids():
    claim = make(cpt="99215", icd10=["Z00.00"], units=2, billed_amount=1000.0)
    ids = [f.flag_id for f in rule_flags(claim, [claim])]
    assert ids == [f for f in RULE_FLAG_IDS if f in ids] and len(ids) >= 3


def _all_claims():
    return [parse_claim(c) for c in json.loads((ROOT / "data" / "v2" / "claims.json").read_text(encoding="utf-8"))]


def test_every_flag_lists_itself_and_only_real_criteria_from_catalogued_policies():
    claims = _all_claims()
    policies_by_flag = {f["flag_id"]: set(f["policy_ids"]) for f in load_catalog()["flags"]}
    seen = set()
    for claim in claims:
        for flag in rule_flags(claim, claims):
            seen.add(flag.flag_id)
            assert flag.detail
            assert flag.evidence[0] == {"kind": "flag", "id": flag.flag_id}
            for ev in flag.evidence:
                if ev["kind"] == "policy":
                    chunk = get_criterion(ev["id"])
                    assert chunk is not None, ev
                    assert chunk.policy_id in policies_by_flag[flag.flag_id], (flag.flag_id, ev)
    assert seen == set(RULE_FLAG_IDS)


# -------------------------------------------------- consistency with the labels
def test_rule_flags_match_expected_flags_on_all_60_claims():
    claims = _all_claims()
    gts = {g["claim_id"]: g for g in
           json.loads((ROOT / "data" / "v2" / "ground_truth.json").read_text(encoding="utf-8"))}
    assert len(claims) == 60
    disagreements = []
    for claim in claims:
        got = {f.flag_id for f in rule_flags(claim, claims)}
        want = {f for f in gts[claim.claim_id]["expected_flags"] if f.startswith("R_")}
        if got != want:
            disagreements.append((claim.claim_id, sorted(got), sorted(want)))
    assert disagreements == []


def test_tables_are_shared_not_reloaded_per_call():
    assert load_tables() is load_tables()
