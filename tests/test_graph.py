import json
import subprocess
import sys
from pathlib import Path

from claims_review.citations import check_explanation
from claims_review.graph import (JUDGE_FAILED_SUFFIX, JUDGE_FAILED_TEXT, NO_ISSUES_TEXT, NODES, build_store,
                                 parse_judge_response, run_claim)
from claims_review.parsing import parse_claim
from conftest import FakeLLM
from factories import raw

ROOT = Path(__file__).resolve().parent.parent

NO_FLAGS = json.dumps({"flags": []})
EM_CLAIM = raw(claim_id="T-EM", cpt="99215", icd10=["Z00.00"], billed_amount=250.0)
KNEE_NOTE = "Right knee pain for two days after a minor twist. No therapy tried yet. No red flags."
KNEE_CLAIM = raw(claim_id="T-KNEE", cpt="73721", modifiers=["RT"], icd10=["M25.561"], billed_amount=450.0,
                 clinical_note=KNEE_NOTE)


def judged(**over):
    base = {"flag_id": "P_NO_CONSERVATIVE_THERAPY", "criterion_id": "POL-003.2",
            "note_quote": "No therapy tried yet", "reason": "The note documents no therapy yet."}
    base.update(over)
    return base


def judge_reply(*flags):
    return json.dumps({"flags": list(flags)})


def run(claim, *replies):
    llm = FakeLLM(*replies)
    return run_claim(claim, llm, build_store([claim])), llm


def evidence_ids(result):
    return {e["id"] for f in result["flags"] for e in f["evidence"] if e["kind"] == "policy"}


# ------------------------------------------------------------------ normal paths
def test_clean_claim_passes_with_one_llm_call_and_the_fixed_text():
    result, llm = run(raw(claim_id="T-CLEAN"), NO_FLAGS)
    assert (result["status"], result["recommended_action"]) == ("ok", "pass")
    assert result["flags"] == [] and result["errors"] == [] and result["rejected"] == []
    assert len(llm.calls) == 1
    assert result["explanation"] == NO_ISSUES_TEXT == "No issues found by rule checks or policy reading."
    assert result["explanation_source"] == "fixed" and result["model"] == "fake-llm"


def test_rule_flag_claim_is_flagged_and_explanation_cites_only_verified_evidence():
    reply = "A level 5 visit with a routine exam diagnosis [POL-002.2] [POL-002.4]."
    result, llm = run(EM_CLAIM, NO_FLAGS, reply)
    assert result["recommended_action"] == "flag" and len(llm.calls) == 2
    [flag] = result["flags"]
    assert flag["flag_id"] == "R_EM_LEVEL_VS_DX" and flag["source"] == "rule" and flag["detail"]
    assert result["explanation"] == reply and result["explanation_source"] == "llm"
    assert check_explanation(result["explanation"], evidence_ids(result)) == []


def test_valid_judged_flag_is_verified_with_policy_and_note_evidence():
    result, llm = run(KNEE_CLAIM, judge_reply(judged()), "No conservative therapy is documented [POL-003.2].")
    assert "POL-003" in result["retrieved_policies"]
    [flag] = result["flags"]
    assert flag["source"] == "policy_reading" and flag["flag_id"] == "P_NO_CONSERVATIVE_THERAPY"
    assert flag["evidence"] == [{"kind": "flag", "id": "P_NO_CONSERVATIVE_THERAPY"},
                                {"kind": "policy", "id": "POL-003.2"},
                                {"kind": "note", "id": "No therapy tried yet"}]
    assert result["rejected"] == [] and result["recommended_action"] == "flag"


def test_judge_prompt_has_the_note_policy_text_with_ids_and_allowed_flags():
    _, llm = run(KNEE_CLAIM, NO_FLAGS)
    text = "\n".join(m["content"] for m in llm.calls[0])
    assert KNEE_NOTE in text and "POL-003.1" in text and "POL-003.4" in text
    assert "P_NO_CONSERVATIVE_THERAPY" in text and "P_DEGENERATIVE_ONLY" in text
    assert "R_UNITS_EXCEEDED" not in text


# ------------------------------------------------------------------ rejected citations
def test_judge_citing_a_criterion_outside_the_retrieved_text_is_rejected_and_traced():
    result, _ = run(KNEE_CLAIM, judge_reply(judged(criterion_id="POL-999.1")))
    assert result["flags"] == [] and result["recommended_action"] == "pass"
    [rej] = result["rejected"]
    assert "not in the retrieved policy text" in rej["reason"] and rej["flag"]["criterion_id"] == "POL-999.1"
    verify = next(t for t in result["trace"] if t["node"] == "verify")
    assert verify["data"]["rejected"] == result["rejected"]


def test_note_quote_not_in_the_note_is_rejected():
    result, _ = run(KNEE_CLAIM, judge_reply(judged(note_quote="patient refused physical therapy")))
    assert result["flags"] == [] and "verbatim" in result["rejected"][0]["reason"]


def test_flag_id_not_a_policy_reading_flag_is_rejected():
    result, _ = run(KNEE_CLAIM, judge_reply(judged(flag_id="R_UNITS_EXCEEDED")))
    assert result["flags"] == [] and "policy_reading flag" in result["rejected"][0]["reason"]


# ------------------------------------------------------------------ failures and retries
def test_invalid_json_twice_keeps_rule_flags_and_never_passes():
    result, llm = run(raw(claim_id="T-CLEAN"), "not json", "{still not json")
    assert result["status"] == "llm_failed" and result["recommended_action"] == "needs_human_review"
    assert result["flags"] == [] and len(llm.calls) == 2 and len(result["errors"]) == 2
    assert result["explanation"] == JUDGE_FAILED_TEXT and result["explanation_source"] == "fixed"

    result, llm = run(EM_CLAIM, "not json", "still not json", "Routine exam billed as level 5 [POL-002.2].")
    assert result["status"] == "llm_failed" and result["recommended_action"] == "flag"
    assert [f["flag_id"] for f in result["flags"]] == ["R_EM_LEVEL_VS_DX"]
    assert result["explanation"].endswith(JUDGE_FAILED_SUFFIX)


def test_invalid_json_once_then_valid_is_ok_and_the_failure_is_recorded():
    result, llm = run(raw(claim_id="T-CLEAN"), "oops", NO_FLAGS)
    assert result["status"] == "ok" and result["recommended_action"] == "pass" and len(llm.calls) == 2
    assert len(result["errors"]) == 1 and result["errors"][0].startswith("judge attempt 1")
    retry = llm.calls[1]
    assert retry[-1]["role"] == "user" and "not valid JSON" in retry[-1]["content"]
    assert retry[-2] == {"role": "assistant", "content": "oops"}
    judge = next(t for t in result["trace"] if t["node"] == "judge")
    assert [a["ok"] for a in judge["data"]["attempts"]] == [False, True]


def test_wrong_json_shape_counts_as_invalid():
    for bad in ('{"flags": "none"}', "[]", '{"flags": [{"flag_id": "x"}]}', '{"flags": [5]}'):
        result, _ = run(raw(claim_id="T-CLEAN"), bad, NO_FLAGS)
        assert result["status"] == "ok" and len(result["errors"]) == 1, bad


def test_an_llm_exception_is_a_failed_attempt_not_a_crash():
    result, llm = run(raw(claim_id="T-CLEAN"), RuntimeError("timeout"), RuntimeError("timeout"))
    assert result["status"] == "llm_failed" and result["recommended_action"] == "needs_human_review"
    assert "RuntimeError" in result["errors"][0] and len(llm.calls) == 2


def test_explanation_with_an_unverified_id_is_retried_then_falls_back_to_the_template():
    bad = "This breaks [POL-009.9]."
    result, llm = run(EM_CLAIM, NO_FLAGS, bad, bad)
    assert len(llm.calls) == 3 and result["explanation_source"] == "template"
    assert result["status"] == "ok" and result["recommended_action"] == "flag"
    assert "POL-009.9" in llm.calls[2][-1]["content"]
    assert check_explanation(result["explanation"], evidence_ids(result)) == []
    assert "R_EM_LEVEL_VS_DX" in result["explanation"] and len(result["errors"]) == 2


def test_explanation_retry_that_fixes_the_ids_is_used():
    result, llm = run(EM_CLAIM, NO_FLAGS, "Bad [POL-009.9].", "Fixed [POL-002.2].")
    assert result["explanation"] == "Fixed [POL-002.2]." and result["explanation_source"] == "llm"
    assert len(llm.calls) == 3


def test_unparseable_claim_is_a_parse_error_with_no_llm_call():
    result, llm = run(raw(claim_id="T-BAD", units=0))
    assert result["status"] == "parse_error" and result["recommended_action"] == "needs_human_review"
    assert llm.calls == [] and result["flags"] == [] and "units" in result["errors"][0]
    assert result["claim_id"] == "T-BAD" and [t["node"] for t in result["trace"]] == ["parse", "finalize"]
    for not_a_claim in ({"no": "id"}, "text"):
        result, llm = run(not_a_claim)
        assert result["status"] == "parse_error" and llm.calls == []


# ------------------------------------------------------------------ shape and determinism
def test_trace_has_every_node_in_order_and_is_json_serializable():
    result, _ = run(EM_CLAIM, NO_FLAGS, "Level 5 with a routine diagnosis [POL-002.2].")
    assert [t["node"] for t in result["trace"]] == list(NODES)
    assert all(set(t) == {"node", "summary", "data"} and t["summary"] for t in result["trace"])
    assert json.loads(json.dumps(result)) == result
    assert set(result) == {"claim_id", "status", "recommended_action", "flags", "explanation", "explanation_source",
                           "rejected", "retrieved_policies", "errors", "model", "trace"}


def test_same_fake_responses_give_identical_results():
    first, _ = run(KNEE_CLAIM, judge_reply(judged(), judged(note_quote="nope")), "Therapy missing [POL-003.2].")
    second, _ = run(KNEE_CLAIM, judge_reply(judged(), judged(note_quote="nope")), "Therapy missing [POL-003.2].")
    assert first == second


def test_cross_claim_rule_uses_the_store_passed_in():
    earlier = raw(claim_id="T-A", date_of_service="2026-02-01")
    later = raw(claim_id="T-B", date_of_service="2026-02-05")
    llm = FakeLLM(NO_FLAGS, "Possible duplicate [POL-004.1].")
    result = run_claim(later, llm, build_store([earlier, later]))
    assert [f["flag_id"] for f in result["flags"]] == ["R_DUPLICATE_CLAIM"]
    llm = FakeLLM(NO_FLAGS)
    assert run_claim(later, llm, build_store([later]))["flags"] == []


def test_build_store_skips_claims_that_do_not_parse():
    store = build_store([raw(claim_id="T-A"), raw(claim_id="T-X", units=0), "junk"])
    assert [c.claim_id for c in store] == ["T-A"]
    assert store[0] == parse_claim(raw(claim_id="T-A"))


def test_parse_judge_response_accepts_a_fenced_block_and_an_empty_list():
    assert parse_judge_response("```json\n" + NO_FLAGS + "\n```") == []
    assert parse_judge_response(judge_reply(judged()))[0]["flag_id"] == "P_NO_CONSERVATIVE_THERAPY"


# ------------------------------------------------------------------ the live script's guards
def _script(*args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "review_claim.py"), *args],
                          cwd=ROOT, capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})


def test_review_claim_refuses_the_test_split_without_final():
    out = _script("--claim", "V2-0001", "--split", "test")
    assert out.returncode == 2 and "--final" in out.stderr and out.stdout == ""


def test_review_claim_unknown_claim_in_the_split_stops_before_any_llm_setup():
    out = _script("--claim", "NOPE-1", "--split", "dev")
    assert out.returncode == 2 and "not in the dev split" in out.stderr and out.stdout == ""
