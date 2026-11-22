"""Phase 6a eval runner: metric math, error counting, test-split guards and output schema.

Every test uses tiny hand-made claims and FakeLLM. No real claim or ground truth is read, and the
guard tests use files under pytest's tmp_path, never data/v2 or results/.
"""
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from claims_review.evaluation import (BASELINE_LABEL, SUMMARY_KEYS, EvalGuardError, build_output, check_guards,
                                      flag_scores, prediction, run_split, score_results, split_records, summarize)
from claims_review.evaluation import test_split_hash as split_hash  # alias so pytest does not collect it
from claims_review.graph import build_store, run_claim
from conftest import FakeLLM
from factories import raw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import generate_claims_v2 as gen  # noqa: E402

DETECTION = {"R_A": "rule", "R_B": "rule", "P_X": "policy_reading", "P_Y": "policy_reading"}
NO_FLAGS = json.dumps({"flags": []})


def gt(cid, flags=(), split="dev"):
    return {"claim_id": cid, "split": split, "is_anomalous": bool(flags), "expected_flags": list(flags)}


def res(cid, flags=(), status="ok", action=None, **extra):
    if action is None:
        action = "flag" if flags else "pass"
    out = {"claim_id": cid, "status": status, "recommended_action": action,
           "flags": [{"flag_id": f} for f in flags], "rejected": [], "trace": [], "explanation_source": "fixed"}
    out.update(extra)
    return out


# ------------------------------------------------------------------ flag metric math
def test_micro_flag_scores_on_a_hand_made_case():
    expected = {"A": {"R_A", "P_X"}, "B": set(), "C": {"R_B"}}
    predicted = {"A": {"R_A"}, "B": {"P_Y"}, "C": None}  # C failed: its flag is a false negative
    s = flag_scores(expected, predicted)
    assert (s["tp"], s["fp"], s["fn"]) == (1, 1, 2)
    assert s["precision"] == 0.5 and s["recall"] == pytest.approx(1 / 3)


def test_subset_scores_use_the_catalog_detection_of_the_flag_id():
    expected = {"A": {"R_A", "P_X"}, "B": set(), "C": {"R_B"}}
    predicted = {"A": {"R_A"}, "B": {"P_Y"}, "C": None}
    pol = flag_scores(expected, predicted, lambda f: DETECTION.get(f) == "policy_reading")
    rule = flag_scores(expected, predicted, lambda f: DETECTION.get(f) == "rule")
    assert (pol["tp"], pol["fp"], pol["fn"], pol["precision"], pol["recall"]) == (0, 1, 1, 0.0, 0.0)
    assert (rule["tp"], rule["fp"], rule["fn"], rule["precision"], rule["recall"]) == (1, 0, 1, 1.0, 0.5)


def test_empty_case_gives_undefined_not_zero():
    s = flag_scores({}, {})
    assert (s["tp"], s["fp"], s["fn"], s["precision"], s["recall"]) == (0, 0, 0, None, None)


def test_zero_denominators():
    # Only clean claims and nothing raised: precision and recall are both undefined.
    s = flag_scores({"A": set(), "B": set()}, {"A": set(), "B": set()})
    assert s["precision"] is None and s["recall"] is None
    # A false alarm with nothing expected: precision is 0, recall is still undefined.
    s = flag_scores({"A": set()}, {"A": {"R_A"}})
    assert s["precision"] == 0.0 and s["recall"] is None
    # Expected flags but nothing raised: recall 0, precision undefined.
    s = flag_scores({"A": {"R_A"}}, {"A": set()})
    assert s["precision"] is None and s["recall"] == 0.0


def test_claim_missing_from_predicted_counts_as_missed():
    s = flag_scores({"A": {"R_A"}}, {})
    assert (s["tp"], s["fn"]) == (0, 1)


# ------------------------------------------------------------------ prediction and errors
@pytest.mark.parametrize("result, expected", [
    (res("A", ["R_A"]), "Flagged"),
    (res("A"), "Clear"),
    (res("A", ["R_A"], status="llm_failed", action="flag"), "Error"),
    (res("A", status="llm_failed", action="needs_human_review"), "Error"),
    (res("A", status="parse_error", action="needs_human_review"), "Error"),
    (res("A", status="exception", action=None), "Error"),
    (res("A", status="ok", action="needs_human_review"), "Error"),
])
def test_prediction(result, expected):
    assert prediction(result) == expected


def test_errors_are_counted_never_dropped_and_count_as_wrong():
    gts = {"A": gt("A", ["R_A"]), "B": gt("B"), "C": gt("C", ["P_X"]), "D": gt("D")}
    results = [res("A", ["R_A"], status="llm_failed", action="flag"),  # rule flag kept but status failed
               res("B", status="parse_error", action="needs_human_review"),
               res("C", ["P_X"]),
               res("D")]
    s = score_results(results, gts, DETECTION)
    assert s["claims"] == 4 and s["errors"] == 2
    assert s["error_claim_ids"] == ["A", "B"]
    assert s["decision_confusion"] == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}
    assert s["decision_accuracy"] == 0.5
    assert s["flag_counts"] == {"tp": 1, "fp": 0, "fn": 1}
    assert s["rule_recall"] == 0.0 and s["rule_precision"] is None
    assert s["policy_reading_recall"] == 1.0 and s["policy_reading_precision"] == 1.0
    assert s["per_claim"] == {"A": {"missed": ["R_A"], "false_positive": []}}


def test_a_missing_result_is_an_error_in_the_scorer_not_a_silent_drop():
    with pytest.raises(ValueError, match="no result"):
        score_results([res("A")], {"A": gt("A"), "B": gt("B")}, DETECTION)


def test_rejected_retries_and_template_fallbacks_are_counted():
    trace = [{"node": "judge", "summary": "", "data": {"attempts": [{"attempt": 1}, {"attempt": 2}]}},
             {"node": "explain", "summary": "", "data": {"attempts": [{"attempt": 1}, {"attempt": 2}]}},
             {"node": "verify", "summary": "", "data": {"attempts": [1, 2, 3]}}]  # not an LLM node
    results = [res("A", ["R_A"], rejected=[{"x": 1}, {"x": 2}], trace=trace, explanation_source="template"),
               res("B", rejected=[{"x": 3}], explanation_source="llm")]
    s = score_results(results, {"A": gt("A", ["R_A"]), "B": gt("B")}, DETECTION)
    assert (s["judge_flags_rejected"], s["retries_used"], s["template_fallbacks"]) == (3, 2, 1)


def test_false_positives_are_listed_per_claim_with_flag_ids_only():
    s = score_results([res("A", ["R_A", "P_Y"])], {"A": gt("A", ["R_A"])}, DETECTION)
    assert s["per_claim"] == {"A": {"missed": [], "false_positive": ["P_Y"]}}


def test_run_split_turns_exceptions_into_error_results_and_reports_progress():
    seen = []

    def review(claim):
        if claim["claim_id"] == "B":
            raise RuntimeError("boom")
        return res(claim["claim_id"])

    runs = run_split([{"claim_id": "A"}, {"claim_id": "B"}], review, 2, progress=lambda *a: seen.append(a))
    assert len(runs) == 2 and all(len(r) == 2 for r in runs)
    assert runs[0][1]["status"] == "exception" and "boom" in runs[0][1]["errors"][0]
    assert seen == [(1, "A", "ok"), (1, "B", "exception"), (2, "A", "ok"), (2, "B", "exception")]
    s = score_results(runs[0], {"A": gt("A"), "B": gt("B", ["R_A"])}, DETECTION)
    assert s["errors"] == 1 and s["error_claim_ids"] == ["B"]
    with pytest.raises(ValueError):
        run_split([], review, 0)


def test_run_split_with_fake_llm_through_the_real_agent():
    clean = raw(claim_id="T-CLEAN", patient_ref="P-1")
    em = raw(claim_id="T-EM", patient_ref="P-2", cpt="99215", icd10=["Z00.00"], billed_amount=250.0)
    llm = FakeLLM(NO_FLAGS, NO_FLAGS, "A level 5 visit with a routine exam diagnosis [POL-002.2] [POL-002.4].")
    store = build_store([clean, em])
    [results] = run_split([clean, em], lambda c: run_claim(c, llm, store), 1)
    s = score_results(results, {"T-CLEAN": gt("T-CLEAN"), "T-EM": gt("T-EM", ["R_EM_LEVEL_VS_DX"])},
                      {"R_EM_LEVEL_VS_DX": "rule"})
    assert s["decision_accuracy"] == 1.0 and s["errors"] == 0
    assert s["rule_precision"] == 1.0 and s["rule_recall"] == 1.0
    assert s["policy_reading_precision"] is None and s["policy_reading_recall"] is None
    assert llm.replies == []


def test_judge_failure_through_the_agent_counts_as_an_error():
    clean = raw(claim_id="T-CLEAN")
    llm = FakeLLM("not json", "still not json")
    [results] = run_split([clean], lambda c: run_claim(c, llm, build_store([clean])), 1)
    s = score_results(results, {"T-CLEAN": gt("T-CLEAN")}, {})
    assert results[0]["status"] == "llm_failed"
    assert s["errors"] == 1 and s["decision_accuracy"] == 0.0 and s["retries_used"] == 1


# ------------------------------------------------------------------ summary across runs
def test_summarize_mean_min_max_and_undefined_values():
    runs = [{k: None for k in SUMMARY_KEYS} for _ in range(3)]
    for r, acc in zip(runs, (0.5, 1.0, 0.75)):
        r["decision_accuracy"] = acc
    runs[0]["flag_precision"] = 0.4
    out = summarize(runs)
    assert set(out) == set(SUMMARY_KEYS)
    assert out["decision_accuracy"] == {"mean": 0.75, "min": 0.5, "max": 1.0, "n_defined": 3}
    assert out["flag_precision"] == {"mean": 0.4, "min": 0.4, "max": 0.4, "n_defined": 1}
    assert out["flag_recall"] == {"mean": None, "min": None, "max": None, "n_defined": 0}


# ------------------------------------------------------------------ guards (tmp fixtures only)
def _fixture(tmp_path, *, frozen=None):
    claims = [raw(claim_id="D-1", patient_ref="P-1"), raw(claim_id="T-1", patient_ref="P-2"),
              raw(claim_id="T-2", patient_ref="P-3")]
    gts = [gt("D-1"), gt("T-1", split="test"), gt("T-2", split="test")]
    data = tmp_path / "data"
    data.mkdir()
    (data / "claims.json").write_text(json.dumps(claims))
    (data / "ground_truth.json").write_text(json.dumps(gts))
    sha = data / "TEST_SPLIT.sha256"
    sha.write_text((frozen if frozen is not None else split_hash(claims, gts)) + "\n")
    results = tmp_path / "results"
    results.mkdir()
    return claims, gts, data, sha, results


def test_hash_matches_the_generator_payload_on_a_synthetic_fixture(tmp_path):
    claims, gts, *_ = _fixture(tmp_path)
    assert split_hash(claims, gts) == gen.test_split_hash(claims, gts)


def test_dev_runs_without_final_and_refuses_final(tmp_path):
    check_guards("dev", False, tmp_path / "eval_dev.json", tmp_path / "eval_test.json")
    with pytest.raises(EvalGuardError, match="--final"):
        check_guards("dev", True, tmp_path / "eval_dev.json", tmp_path / "eval_test.json")


def test_test_split_without_final_is_refused_before_loading_anything(tmp_path):
    def loader():
        raise AssertionError("records must not be loaded")
    with pytest.raises(EvalGuardError, match="without --final"):
        check_guards("test", False, tmp_path / "eval_test.json", tmp_path / "eval_test.json", loader, tmp_path / "x")


def test_final_must_write_the_fixed_path(tmp_path):
    claims, gts, _, sha, results = _fixture(tmp_path)
    with pytest.raises(EvalGuardError, match="must write"):
        check_guards("test", True, results / "other.json", results / "eval_test.json", lambda: (claims, gts), sha)


def test_final_refuses_when_the_final_file_exists_and_does_not_touch_it(tmp_path):
    claims, gts, _, sha, results = _fixture(tmp_path)
    final = results / "eval_test.json"
    final.write_text("earlier final run")
    with pytest.raises(EvalGuardError, match="already exists"):
        check_guards("test", True, final, final, lambda: (claims, gts), sha)
    assert final.read_text() == "earlier final run"


def test_final_refuses_on_hash_mismatch_and_never_writes_the_hash_file(tmp_path):
    claims, gts, _, sha, results = _fixture(tmp_path, frozen="0" * 64)
    before = sha.read_bytes()
    with pytest.raises(EvalGuardError, match="does not match"):
        check_guards("test", True, results / "eval_test.json", results / "eval_test.json", lambda: (claims, gts), sha)
    assert sha.read_bytes() == before


def test_final_with_matching_hash_passes(tmp_path):
    claims, gts, _, sha, results = _fixture(tmp_path)
    check_guards("test", True, results / "eval_test.json", results / "eval_test.json", lambda: (claims, gts), sha)


def test_changed_test_claim_changes_the_hash(tmp_path):
    claims, gts, *_ = _fixture(tmp_path)
    before = split_hash(claims, gts)
    claims[1]["units"] = 2
    assert split_hash(claims, gts) != before
    claims[1]["units"] = 1
    claims[0]["units"] = 2  # a dev claim is not in the hash
    assert split_hash(claims, gts) == before


def _load_script():
    spec = importlib.util.spec_from_file_location("run_eval_script", ROOT / "scripts" / "run_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_final_run_happens_once_on_tmp_fixtures(tmp_path, capsys):
    _, _, data, sha, results = _fixture(tmp_path)
    script = _load_script()
    sha_before = sha.read_bytes()
    llms = []

    def factory():
        llms.append(FakeLLM(NO_FLAGS, NO_FLAGS))
        return llms[-1]

    argv = ["--split", "test", "--final", "--runs", "1", "--out", str(results / "eval_test.json")]
    assert script.main(argv, data_dir=data, results_dir=results, llm_factory=factory) == 0
    out = json.loads((results / "eval_test.json").read_text())
    assert out["split"] == "test" and out["claim_count"] == 2 and out["runs"] == 1
    printed = capsys.readouterr().out.splitlines()
    assert [line for line in printed if line.startswith("run ")] == ["run 1 T-1 ok", "run 1 T-2 ok"]
    # Second final run is refused and the first file is untouched.
    first = (results / "eval_test.json").read_text()
    assert script.main(argv, data_dir=data, results_dir=results, llm_factory=factory) == 2
    assert (results / "eval_test.json").read_text() == first and len(llms) == 1
    assert sha.read_bytes() == sha_before


def test_script_refuses_test_without_final_and_builds_no_llm(tmp_path):
    _, _, data, _, results = _fixture(tmp_path)
    script = _load_script()
    built = []
    assert script.main(["--split", "test"], data_dir=data, results_dir=results,
                       llm_factory=lambda: built.append(1)) == 2
    assert built == [] and not any(results.iterdir())


def test_script_dev_run_uses_only_dev_claims(tmp_path, capsys):
    _, _, data, _, results = _fixture(tmp_path)
    script = _load_script()
    out_path = results / "eval_dev.json"
    assert script.main(["--runs", "2", "--out", str(out_path)], data_dir=data, results_dir=results,
                       llm_factory=lambda: FakeLLM(NO_FLAGS, NO_FLAGS)) == 0
    printed = [line for line in capsys.readouterr().out.splitlines() if line.startswith("run ")]
    assert printed == ["run 1 D-1 ok", "run 2 D-1 ok"]
    out = json.loads(out_path.read_text())
    assert out["split"] == "dev" and out["claim_count"] == 1 and out["runs"] == 2


# ------------------------------------------------------------------ output schema
def test_output_schema():
    claims = [raw(claim_id="A", patient_ref="P-1"), raw(claim_id="B", patient_ref="P-2", billed_amount=500.0)]
    gts_list = [gt("A"), gt("B", ["R_PRICE_OUTLIER"])]
    _, gts = split_records(claims, gts_list, "dev")
    runs = [[res("A"), res("B", ["R_PRICE_OUTLIER"])],
            [res("A", ["P_X"]), res("B", status="llm_failed", action="needs_human_review")]]
    now = datetime(2026, 11, 22, 12, 0, 1, 500, tzinfo=timezone.utc)
    out = build_output(split="dev", model="fake-llm", temperature=0, runs_results=runs, gts=gts, claims=claims,
                       detection={"R_PRICE_OUTLIER": "rule", "P_X": "policy_reading"}, commit="abc123", now=now)
    json.dumps(out)  # serializable
    assert set(out) == {"schema_version", "split", "claim_count", "runs", "model", "temperature", "date_utc",
                        "git_commit", "agent", "rules_only_baseline"}
    assert (out["split"], out["claim_count"], out["runs"], out["model"], out["temperature"], out["git_commit"]) == \
        ("dev", 2, 2, "fake-llm", 0, "abc123")
    assert out["date_utc"] == "2026-11-22T12:00:01+00:00"
    assert len(out["agent"]["per_run"]) == 2
    for run in out["agent"]["per_run"]:
        assert set(SUMMARY_KEYS) <= set(run)
        for cid, miss in run["per_claim"].items():
            assert set(miss) == {"missed", "false_positive"} and cid in gts
    assert out["agent"]["per_run"][1]["errors"] == 1
    assert out["agent"]["summary"]["decision_accuracy"] == {"mean": 0.5, "min": 0.0, "max": 1.0, "n_defined": 2}
    base = out["rules_only_baseline"]
    assert base["label"] == BASELINE_LABEL == "at ceiling by construction (D22)"
    assert base["decision_accuracy"] == 1.0 and base["rule_recall"] == 1.0 and base["errors"] == 0
    assert "retries_used" not in base
