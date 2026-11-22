import json
from pathlib import Path

import pytest

from claims_review.citations import check_explanation, template_explanation, verify_judge_flags
from claims_review.models import Flag
from claims_review.parsing import parse_claim
from claims_review.retrieval import retrieve_policies
from claims_review.rules import load_catalog, rule_flags
from factories import make

ROOT = Path(__file__).resolve().parent.parent

CATALOG = load_catalog()
POLICIES = retrieve_policies("knee MRI conservative therapy six weeks physical therapy", k=2)
NOTE = "Right knee pain for two days.\nNo therapy   tried yet. No red flags."
CLAIM = make(cpt="73721", icd10=["M25.561"], clinical_note=NOTE)


def judged(**over):
    base = {"flag_id": "P_NO_CONSERVATIVE_THERAPY", "criterion_id": "POL-003.2",
            "note_quote": "No therapy tried yet", "reason": "No therapy documented."}
    base.update(over)
    return base


def test_retrieved_policies_used_here_contain_pol_003():
    assert "POL-003" in [p.policy_id for p in POLICIES]


def test_valid_flag_is_verified_and_whitespace_in_the_note_is_collapsed():
    verified, rejected = verify_judge_flags([judged()], CLAIM, POLICIES, CATALOG)
    assert rejected == [] and [v["flag_id"] for v in verified] == ["P_NO_CONSERVATIVE_THERAPY"]
    assert verified[0]["note_quote"] == "No therapy tried yet"


def test_quote_spanning_a_line_break_matches_after_collapsing():
    verified, _ = verify_judge_flags([judged(note_quote="two days. No therapy")], CLAIM, POLICIES, CATALOG)
    assert len(verified) == 1


def test_empty_flag_list_gives_nothing():
    assert verify_judge_flags([], CLAIM, POLICIES, CATALOG) == ([], [])


def test_flag_id_must_be_a_policy_reading_flag_in_the_catalog():
    for bad in ("R_UNITS_EXCEEDED", "P_MADE_UP", ""):
        verified, rejected = verify_judge_flags([judged(flag_id=bad)], CLAIM, POLICIES, CATALOG)
        assert verified == [] and "policy_reading flag" in rejected[0]["reason"]


def test_criterion_id_must_be_in_the_retrieved_text():
    retrieved_ids = {p.policy_id for p in POLICIES}
    other = next(pid for pid in ("POL-001", "POL-002", "POL-004", "POL-005", "POL-006", "POL-007", "POL-008")
                 if pid not in retrieved_ids)
    for bad in ("POL-999.1", f"{other}.1", "POL-003.9", "POL-003"):
        verified, rejected = verify_judge_flags([judged(criterion_id=bad)], CLAIM, POLICIES, CATALOG)
        assert verified == [] and "not in the retrieved policy text" in rejected[0]["reason"], bad


def test_note_quote_must_be_a_verbatim_non_empty_substring():
    for bad in ("never tried therapy", "no therapy tried yet", "No therapy tried yet!"):
        verified, rejected = verify_judge_flags([judged(note_quote=bad)], CLAIM, POLICIES, CATALOG)
        assert verified == [] and "verbatim" in rejected[0]["reason"], bad
    for empty in ("", "   "):
        verified, rejected = verify_judge_flags([judged(note_quote=empty)], CLAIM, POLICIES, CATALOG)
        assert verified == [] and "empty" in rejected[0]["reason"]


def test_duplicate_flag_id_keeps_the_first_and_rejects_the_second():
    first, second = judged(), judged(criterion_id="POL-003.1")
    verified, rejected = verify_judge_flags([first, second], CLAIM, POLICIES, CATALOG)
    assert [v["criterion_id"] for v in verified] == ["POL-003.2"]
    assert rejected == [{"flag": second, "reason": "duplicate flag_id 'P_NO_CONSERVATIVE_THERAPY'"}]


def test_a_rejected_first_flag_does_not_block_a_valid_second_one():
    bad, good = judged(note_quote="not in note"), judged()
    verified, rejected = verify_judge_flags([bad, good], CLAIM, POLICIES, CATALOG)
    assert len(verified) == 1 and len(rejected) == 1


def test_malformed_items_are_rejected_not_raised():
    items = ["text", None, {"flag_id": "P_NO_CONSERVATIVE_THERAPY"}, judged(reason=5)]
    verified, rejected = verify_judge_flags(items, CLAIM, POLICIES, CATALOG)
    assert verified == [] and len(rejected) == 4 and all(r["reason"] for r in rejected)


def test_check_explanation_accepts_allowed_ids_and_none():
    assert check_explanation("Level too high [POL-002.2] [POL-002.4].", {"POL-002.2", "POL-002.4"}) == []
    assert check_explanation("No ids at all.", set()) == []


def test_check_explanation_lists_each_unverified_id_once():
    problems = check_explanation("See [POL-009.9] and [POL-009.9] and [POL-002.2].", {"POL-002.2"})
    assert len(problems) == 1 and "POL-009.9" in problems[0]


def test_check_explanation_rejects_empty_text():
    assert check_explanation("   ", set()) and check_explanation("", {"POL-002.2"})


def test_template_explanation_is_deterministic_and_uses_only_evidence_ids():
    flags = [Flag("R_EM_LEVEL_VS_DX", "Level needs tier 3, highest diagnosis tier is 1",
                  [{"kind": "flag", "id": "R_EM_LEVEL_VS_DX"}, {"kind": "policy", "id": "POL-002.2"},
                   {"kind": "policy", "id": "POL-002.4"}]),
             Flag("P_NO_CONSERVATIVE_THERAPY", "No therapy documented; see POL-003.9.",
                  [{"kind": "policy", "id": "POL-003.2"}, {"kind": "note", "id": "No therapy tried yet"}])]
    text = template_explanation(flags)
    assert text == template_explanation(flags)
    assert "[POL-002.2] [POL-002.4]" in text and "[POL-003.2]" in text
    assert "POL-003.9" not in text and "[unverified id]" in text
    assert check_explanation(text, {"POL-002.2", "POL-002.4", "POL-003.2"}) == []


# ------------------------------------------------------------------ flag-to-criterion link (Phase 3b)
def test_criterion_from_a_retrieved_but_unlinked_policy_is_rejected():
    other = next(p for p in POLICIES if p.policy_id != "POL-003")
    cid = other.criteria[0][0]
    verified, rejected = verify_judge_flags([judged(criterion_id=cid)], CLAIM, POLICIES, CATALOG)
    assert verified == []
    assert rejected[0]["reason"] == f"criterion_id '{cid}' is not linked to flag 'P_NO_CONSERVATIVE_THERAPY' in the catalog"


def test_each_flag_is_linked_only_to_its_own_policy():
    knee_note = "Torn meniscus on MRI. Degenerative changes only. No therapy tried yet."
    claim = make(cpt="29881", icd10=["M23.221"], clinical_note=knee_note)
    pols = retrieve_policies("knee arthroscopy meniscus degenerative therapy mri conservative", k=4)
    ids = {p.policy_id for p in pols}
    assert {"POL-001", "POL-003"} <= ids
    ok = judged(flag_id="P_DEGENERATIVE_ONLY", criterion_id="POL-001.1", note_quote="Degenerative changes only")
    wrong = judged(flag_id="P_DEGENERATIVE_ONLY", criterion_id="POL-003.2", note_quote="No therapy tried yet")
    verified, rejected = verify_judge_flags([ok, wrong], claim, pols, CATALOG)
    assert [v["flag_id"] for v in verified] == ["P_DEGENERATIVE_ONLY"]
    assert "not linked to flag 'P_DEGENERATIVE_ONLY'" in rejected[0]["reason"]


def test_existing_valid_cases_still_verify_for_every_criterion_of_the_linked_policy():
    for cid in ("POL-003.1", "POL-003.2", "POL-003.3", "POL-003.4"):
        verified, rejected = verify_judge_flags([judged(criterion_id=cid)], CLAIM, POLICIES, CATALOG)
        assert len(verified) == 1 and rejected == [], cid


# ------------------------------------------------------------------ bare ids and decision wording
def test_bare_policy_id_is_a_problem_unless_an_allowed_id_is_in_that_policy():
    assert check_explanation("Duplicate of an earlier claim, see POL-004.", {"POL-002.2"})
    assert check_explanation("Duplicate of an earlier claim, see POL-004.", {"POL-004.1"}) == []
    assert check_explanation("See POL-004 [POL-004.1].", {"POL-004.1"}) == []
    assert check_explanation("See policy POL-004, criterion [POL-004.1].", {"POL-003.2"})
    assert check_explanation("Cited as [POL-004.1] only.", {"POL-004.1"}) == []


@pytest.mark.parametrize("word", ["denied", "deny", "denies", "approved", "approve", "approves",
                                  "rejected", "reject", "rejects", "Denied", "APPROVED", "Rejects"])
def test_each_decision_word_is_caught_case_insensitively(word):
    problems = check_explanation(f"The claim is {word} because of [POL-002.2].", {"POL-002.2"})
    assert problems == ["uses decision wording; the agent only recommends review"]


@pytest.mark.parametrize("text", ["Needs prior approval review [POL-002.2].", "Disapproved by nobody [POL-002.2].",
                                  "A rejection letter is not mentioned [POL-002.2].", "The undenied fact [POL-002.2]."])
def test_decision_words_inside_other_words_are_allowed(text):
    assert check_explanation(text, {"POL-002.2"}) == []


def test_decision_wording_is_reported_once_even_if_repeated():
    assert len(check_explanation("Denied, denied and rejected.", set())) == 1


def _dev_claims():
    raws = json.loads((ROOT / "data" / "v2" / "claims.json").read_text(encoding="utf-8"))
    split = {g["claim_id"]: g["split"] for g in
             json.loads((ROOT / "data" / "v2" / "ground_truth.json").read_text(encoding="utf-8"))}
    return [parse_claim(r) for r in raws if split[r["claim_id"]] == "dev"]


def test_template_explanation_passes_check_explanation_for_every_dev_claim_with_rule_flags():
    claims = _dev_claims()
    assert len(claims) == 20
    checked = 0
    for claim in claims:
        flags = rule_flags(claim, claims)
        if not flags:
            continue
        allowed = {e["id"] for f in flags for e in f.evidence if e["kind"] == "policy"}
        text = template_explanation(flags)
        assert check_explanation(text, allowed) == [], (claim.claim_id, text)
        checked += 1
    assert checked >= 7


def test_template_removes_decision_words_and_bare_ids_from_a_judged_reason():
    flag = Flag("P_NO_CONSERVATIVE_THERAPY", "Should be denied under POL-003, not POL-004.",
                [{"kind": "policy", "id": "POL-003.2"}])
    text = template_explanation([flag])
    assert "denied" not in text and "POL-004" not in text and "POL-003," in text
    assert check_explanation(text, {"POL-003.2"}) == []
