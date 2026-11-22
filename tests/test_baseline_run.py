import subprocess
import sys
from pathlib import Path

from claims_review.baseline import run_rules_only
from factories import raw

ROOT = Path(__file__).resolve().parent.parent


def test_one_result_per_input_claim_in_order():
    claims = [raw(claim_id="A-1", patient_ref="P-1"),
              raw(claim_id="A-2", patient_ref="P-2", cpt="99216"),
              raw(claim_id="A-3", patient_ref="P-3", billed_amount=500.0)]
    results = run_rules_only(claims)
    assert [r.claim_id for r in results] == ["A-1", "A-2", "A-3"]
    assert [r.decision for r in results] == ["Clear", "Flagged", "Flagged"]
    assert results[0].flags == []
    assert [f.flag_id for f in results[1].flags] == ["R_INVALID_CODE"]
    assert [f.flag_id for f in results[2].flags] == ["R_PRICE_OUTLIER"]


def test_malformed_claims_become_error_results_and_are_never_dropped():
    claims = [raw(claim_id="A-1", patient_ref="P-1"), raw(claim_id="BAD-1", units="three"), {"no": "claim id"},
              "not a dict", raw(claim_id="A-2", patient_ref="P-2")]
    results = run_rules_only(claims)
    assert len(results) == len(claims)
    assert [r.decision for r in results] == ["Clear", "Error", "Error", "Error", "Clear"]
    assert results[1].claim_id == "BAD-1" and "units" in results[1].error
    assert results[2].claim_id == "(index 2)" and results[3].claim_id == "(index 3)"
    assert all(r.flags == [] for r in results if r.decision == "Error")


def test_cross_claim_rules_still_work_when_another_claim_failed():
    claims = [raw(claim_id="A-1", date_of_service="2026-02-01"), raw(claim_id="BAD", units=0),
              raw(claim_id="A-2", date_of_service="2026-02-03")]
    results = {r.claim_id: r for r in run_rules_only(claims)}
    assert results["BAD"].decision == "Error"
    assert results["A-1"].decision == "Clear"
    assert results["A-2"].decision == "Flagged"
    assert [f.flag_id for f in results["A-2"].flags] == ["R_DUPLICATE_CLAIM"]


def test_empty_input_gives_empty_output():
    assert run_rules_only([]) == []


def test_unexpected_exception_inside_rules_becomes_an_error(monkeypatch):
    import claims_review.baseline as baseline

    def boom(claim, store):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(baseline, "rule_flags", boom)
    results = run_rules_only([raw(claim_id="A-1")])
    assert results[0].decision == "Error" and "synthetic failure" in results[0].error


def _run(script, *args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args],
                          cwd=ROOT, capture_output=True, text=True)


def test_run_baseline_refuses_test_split_without_final():
    out = _run("run_baseline.py", "--split", "test")
    assert out.returncode == 2 and "--final" in out.stderr and out.stdout == ""


def test_eval_retrieval_refuses_test_split_without_final():
    out = _run("eval_retrieval.py", "--split", "test")
    assert out.returncode == 2 and "--final" in out.stderr and out.stdout == ""


def test_run_baseline_dev_runs_and_covers_only_dev_claims():
    out = _run("run_baseline.py", "--split", "dev", "--show-misses")
    assert out.returncode == 0, out.stderr
    assert "split=dev: 20 claims" in out.stdout and "PER FLAG" in out.stdout and "MISSES" in out.stdout


def test_eval_retrieval_dev_runs():
    out = _run("eval_retrieval.py", "--split", "dev")
    assert out.returncode == 0, out.stderr
    assert "BM25 RETRIEVAL, split=dev" in out.stdout

