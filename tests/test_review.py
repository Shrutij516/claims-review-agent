import json
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pytest

from claims_review import graph as graph_module
from claims_review.graph import NODES, build_graph, build_store, run_claim
from claims_review.review import (ReviewActionError, ReviewExistsError, close_saver, get_review, open_saver,
                                  start_review, submit_review, validate_review_action)
from conftest import FakeLLM
from factories import raw

ROOT = Path(__file__).resolve().parent.parent
NO_FLAGS = json.dumps({"flags": []})
CLEAN = raw(claim_id="T-CLEAN")
FLAGGED = raw(claim_id="T-EM", cpt="99215", icd10=["Z00.00"], billed_amount=250.0)
EXPLAIN = "A level 5 visit with a routine exam diagnosis was flagged for review [POL-002.2] [POL-002.4]."
APPROVE = {"action": "approve", "reviewer": "alex"}
FIXED_TIME = "2026-11-22T10:00:00+00:00"


@pytest.fixture
def db(tmp_path):
    return tmp_path / "runs" / "checkpoints.sqlite"


@pytest.fixture
def saver(db):
    s = open_saver(db)
    yield s
    try:
        close_saver(s)
    except Exception:
        pass


def start(saver, claim, *replies):
    llm = FakeLLM(*replies)
    return start_review(claim, llm, build_store([claim]), saver), llm


def flagged(saver, claim=FLAGGED):
    return start(saver, claim, NO_FLAGS, EXPLAIN)


def failed_judge(saver):
    return start(saver, CLEAN, "bad json", "still bad")


# ------------------------------------------------------------------ start
def test_start_review_pauses_with_a_payload_that_matches_the_agent_result(saver):
    out, llm = flagged(saver)
    assert out["thread_id"] == "T-EM" and out["status"] == "awaiting_review"
    payload = out["payload"]
    assert set(payload) == {"claim_id", "recommended_action", "flags", "explanation", "errors"}
    expected = run_claim(FLAGGED, FakeLLM(NO_FLAGS, EXPLAIN), build_store([FLAGGED]))
    for key in payload:
        assert payload[key] == expected[key], key
    assert payload["recommended_action"] == "flag" and len(llm.calls) == 2
    assert out["agent_result"] == expected
    assert [t["node"] for t in out["agent_result"]["trace"]] == list(NODES)
    assert get_review("T-EM", saver) == {"status": "awaiting_review", "payload": payload,
                                         "agent_result": out["agent_result"]}


# ------------------------------------------------------------------ actions
def test_approve_a_flag_recommendation(saver):
    flagged(saver)
    record = submit_review("T-EM", APPROVE, saver, now=lambda: FIXED_TIME)
    assert record["final_status"] == "approved" and record["claim_id"] == "T-EM"
    assert record["human"] == {"action": "approve", "final_decision": "flag", "reason": None, "question": None,
                               "reviewer": "alex", "at": FIXED_TIME}


def test_approve_a_pass_recommendation(saver):
    start(saver, CLEAN, NO_FLAGS)
    record = submit_review("T-CLEAN", APPROVE, saver)
    assert record["final_status"] == "approved" and record["human"]["final_decision"] == "pass"


def test_override_pass_to_flag_and_flag_to_pass_store_the_reason(saver):
    start(saver, CLEAN, NO_FLAGS)
    record = submit_review("T-CLEAN", {"action": "override", "reviewer": "alex", "final_decision": "flag",
                                       "reason": "Note mentions a prior denial pattern"}, saver)
    assert record["final_status"] == "overridden"
    assert record["human"]["final_decision"] == "flag" and record["human"]["reason"] == "Note mentions a prior denial pattern"
    assert record["agent_result"]["recommended_action"] == "pass"

    flagged(saver)
    record = submit_review("T-EM", {"action": "override", "reviewer": "alex", "final_decision": "pass",
                                    "reason": "Visit was complex, documented elsewhere"}, saver)
    assert record["final_status"] == "overridden" and record["human"]["final_decision"] == "pass"
    assert record["human"]["reason"] == "Visit was complex, documented elsewhere"


@pytest.mark.parametrize("action", [
    {"action": "override", "reviewer": "alex", "final_decision": "pass"},
    {"action": "override", "reviewer": "alex", "final_decision": "pass", "reason": "   "},
    {"action": "override", "reviewer": "alex", "final_decision": "maybe", "reason": "x"},
    {"action": "override", "reviewer": "alex", "reason": "x"},
    {"action": "approve", "reviewer": ""},
    {"action": "approve", "reviewer": "  "},
    {"action": "approve"},
    {"action": "delete", "reviewer": "alex"},
    {"reviewer": "alex"},
    "approve",
])
def test_invalid_actions_are_rejected_and_the_review_stays_awaiting(saver, action):
    flagged(saver)
    with pytest.raises(ReviewActionError) as err:
        submit_review("T-EM", action, saver)
    assert err.value.problems
    assert get_review("T-EM", saver)["status"] == "awaiting_review"
    assert submit_review("T-EM", APPROVE, saver)["final_status"] == "approved"   # still resumable


def test_approve_is_rejected_when_the_agent_needs_a_human_but_override_works(saver):
    out, _ = failed_judge(saver)
    assert out["payload"]["recommended_action"] == "needs_human_review"
    with pytest.raises(ReviewActionError, match="approve is not allowed"):
        submit_review("T-CLEAN", APPROVE, saver)
    assert get_review("T-CLEAN", saver)["status"] == "awaiting_review"
    record = submit_review("T-CLEAN", {"action": "override", "reviewer": "alex", "final_decision": "pass",
                                       "reason": "Checked the note by hand"}, saver)
    assert record["final_status"] == "overridden" and record["agent_result"]["status"] == "llm_failed"


def test_request_info_needs_a_question_and_ends_as_needs_info(saver):
    flagged(saver)
    with pytest.raises(ReviewActionError, match="question"):
        submit_review("T-EM", {"action": "request_info", "reviewer": "alex"}, saver)
    with pytest.raises(ReviewActionError):
        submit_review("T-EM", {"action": "request_info", "reviewer": "alex", "question": " "}, saver)
    record = submit_review("T-EM", {"action": "request_info", "reviewer": "alex",
                                    "question": "Which diagnosis supports level 5?"}, saver)
    assert record["final_status"] == "needs_info"
    assert record["human"]["question"] == "Which diagnosis supports level 5?" and record["human"]["final_decision"] is None


def test_request_info_is_allowed_even_when_the_agent_needs_a_human(saver):
    failed_judge(saver)
    record = submit_review("T-CLEAN", {"action": "request_info", "reviewer": "alex", "question": "Rerun?"}, saver)
    assert record["final_status"] == "needs_info"


# ------------------------------------------------------------------ persistence and resume
def test_restart_a_new_saver_and_a_fresh_graph_can_finish_the_review(db):
    first = open_saver(db)
    llm = FakeLLM(NO_FLAGS, EXPLAIN)
    started = start_review(FLAGGED, llm, build_store([FLAGGED]), first)
    close_saver(first)
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # no serialization warnings on a checkpoint round trip
        second = open_saver(db)
        try:
            assert get_review("T-EM", second) == {"status": "awaiting_review", "payload": started["payload"],
                                                  "agent_result": started["agent_result"]}
            record = submit_review("T-EM", APPROVE, second, now=lambda: FIXED_TIME)
        finally:
            close_saver(second)
    assert record["final_status"] == "approved" and record["human"]["at"] == FIXED_TIME
    third = open_saver(db)
    try:
        assert get_review("T-EM", third) == {"status": "completed", "record": record}
    finally:
        close_saver(third)


def test_resume_makes_no_llm_call_and_re_runs_no_earlier_node(saver, monkeypatch):
    out, llm = flagged(saver)
    calls_after_start = len(llm.calls)

    def must_not_run(*args, **kwargs):
        raise AssertionError("a node before human_review ran again on resume")

    for name in ("rule_flags", "retrieve_policies", "make_query", "verify_judge_flags", "check_explanation",
                 "template_explanation"):
        monkeypatch.setattr(graph_module, name, must_not_run)
    record = submit_review("T-EM", APPROVE, saver)
    assert len(llm.calls) == calls_after_start == 2
    nodes = [t["node"] for t in record["agent_result"]["trace"]]
    assert nodes == list(NODES) + ["human_review", "record"]
    for key in out["payload"]:
        assert record["agent_result"][key] == out["payload"][key], key


def test_record_has_the_agent_result_the_human_action_and_final_status(saver):
    flagged(saver)
    record = submit_review("T-EM", APPROVE, saver)
    assert set(record) == {"claim_id", "agent_result", "human", "final_status"}
    assert set(record["agent_result"]) == {"claim_id", "status", "recommended_action", "flags", "explanation",
                                           "explanation_source", "rejected", "retrieved_policies", "errors",
                                           "model", "trace"}
    assert set(record["human"]) == {"action", "final_decision", "reason", "question", "reviewer", "at"}
    assert json.loads(json.dumps(record)) == record


def test_one_review_per_claim_and_lookups(saver):
    flagged(saver)
    with pytest.raises(ReviewExistsError, match="already exists"):
        flagged(saver)
    assert get_review("NOPE-1", saver) == {"status": "not_found"}
    record = submit_review("T-EM", APPROVE, saver)
    assert get_review("T-EM", saver) == {"status": "completed", "record": record}
    with pytest.raises(ReviewExistsError):
        flagged(saver)
    with pytest.raises(ReviewActionError, match="no review is waiting"):
        submit_review("T-EM", APPROVE, saver)
    with pytest.raises(ReviewActionError, match="not_found"):
        submit_review("NOPE-1", APPROVE, saver)


def test_reviews_for_different_claims_are_independent(saver):
    flagged(saver)
    start(saver, CLEAN, NO_FLAGS)
    submit_review("T-EM", APPROVE, saver)
    assert get_review("T-CLEAN", saver)["status"] == "awaiting_review"


def test_unparseable_claim_can_be_reviewed_and_never_approved(saver):
    out, llm = start(saver, raw(claim_id="T-BAD", units=0))
    assert llm.calls == [] and out["payload"]["recommended_action"] == "needs_human_review"
    with pytest.raises(ReviewActionError):
        submit_review("T-BAD", APPROVE, saver)
    assert submit_review("T-BAD", {"action": "request_info", "reviewer": "a", "question": "Fix units?"}, saver)


def test_claim_without_an_id_cannot_start_a_review(saver):
    with pytest.raises(ValueError, match="claim_id"):
        start_review({"no": "id"}, FakeLLM(), [], saver)


# ------------------------------------------------------------------ clock and the plain agent
def test_the_clock_makes_the_timestamp_deterministic(saver):
    flagged(saver)
    when = datetime(2026, 11, 22, 9, 30, tzinfo=timezone.utc)
    assert submit_review("T-EM", APPROVE, saver, now=lambda: when)["human"]["at"] == "2026-11-22T09:30:00+00:00"


def test_the_default_clock_gives_a_utc_iso_timestamp(saver):
    flagged(saver)
    at = submit_review("T-EM", APPROVE, saver)["human"]["at"]
    assert datetime.fromisoformat(at).utcoffset().total_seconds() == 0


def test_human_review_false_is_identical_to_omitting_the_flag():
    store = build_store([FLAGGED])
    plain = run_claim(FLAGGED, FakeLLM(NO_FLAGS, EXPLAIN), store)
    explicit = build_graph(FakeLLM(NO_FLAGS, EXPLAIN), store, human_review=False).invoke(
        {"claim_raw": FLAGGED, "errors": [], "trace": []})
    assert explicit["result"] == plain
    assert "record" not in explicit and [t["node"] for t in plain["trace"]] == list(NODES)


def test_human_review_true_needs_a_checkpointer():
    with pytest.raises(ValueError, match="checkpointer"):
        build_graph(FakeLLM(), [], human_review=True)


# ------------------------------------------------------------------ validation function
def test_validate_review_action_accepts_each_valid_form():
    assert validate_review_action(APPROVE, "flag") == [] and validate_review_action(APPROVE, "pass") == []
    assert validate_review_action({"action": "override", "reviewer": "a", "final_decision": "flag",
                                   "reason": "r"}, "needs_human_review") == []
    assert validate_review_action({"action": "request_info", "reviewer": "a", "question": "q"}, "needs_human_review") == []


def test_validate_review_action_lists_every_problem():
    problems = validate_review_action({"action": "override", "reviewer": "", "final_decision": "x"}, "flag")
    assert len(problems) == 3


# ------------------------------------------------------------------ the script's guards
def _script(*args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "human_review.py"), *args],
                          cwd=ROOT, capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})


@pytest.mark.parametrize("command", ["start", "show", "submit"])
def test_human_review_script_refuses_the_test_split_without_final(command):
    extra = ["--action", "approve"] if command == "submit" else []
    out = _script(command, "--claim", "V2-0001", "--split", "test", *extra)
    assert out.returncode == 2 and "--final" in out.stderr and out.stdout == ""


def test_human_review_script_start_stops_for_an_unknown_claim_before_any_llm_setup():
    out = _script("start", "--claim", "NOPE-1", "--split", "dev")
    assert out.returncode == 2 and "not in the dev split" in out.stderr and out.stdout == ""


def test_human_review_script_submit_reports_problems_with_exit_2(tmp_path):
    db = tmp_path / "x.sqlite"
    out = _script("submit", "--claim", "NOPE-1", "--split", "dev", "--action", "approve", "--db", str(db))
    assert out.returncode == 2 and "no review is waiting" in out.stderr and out.stdout == ""
    shown = _script("show", "--claim", "NOPE-1", "--split", "dev", "--db", str(db))
    assert shown.returncode == 0 and json.loads(shown.stdout) == {"status": "not_found"}
