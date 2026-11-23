"""Phase 6 evaluation: score agent results against ground truth, the test-split guards, and the output file.

Everything here is pure and testable with tiny hand-made inputs. The script `scripts/run_eval.py`
does the file reading, the LLM construction and the printing.

Metric definitions (D54):
- decision accuracy: a claim is predicted Flagged when the agent recommends "flag", Clear when it
  recommends "pass", and Error when its status is not "ok" (llm_failed, parse_error) or the run
  raised. Errors count as wrong (D13) and are also reported as a count.
- flag precision and recall are micro over (claim, flag id) pairs. An error claim raises no flag,
  so every flag it should have raised is a false negative. A zero denominator gives None.
- the same precision and recall restricted to policy_reading flag ids and to rule flag ids, chosen
  by the catalog's detection field for the flag id, not by which node raised it.
"""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from claims_review.baseline import run_rules_only
from claims_review.metrics import compute_metrics

BASELINE_LABEL = ("rules only, no LLM. Rule-flag scores are at ceiling by construction (D22); "
                  "it cannot raise policy_reading flags, so its decision accuracy and flag recall are lower.")
SCHEMA_VERSION = 1
ABORT_AFTER_CONSECUTIVE_FAILURES = 3
STREAK_STATUSES = ("llm_failed", "exception")  # statuses that count toward the failure streak (D64)
ERROR_TEXT_LIMIT = 300
KEY_PATTERN = re.compile(r"gsk_[A-Za-z0-9]{20,}")
SUMMARY_KEYS = ("decision_accuracy", "flag_precision", "flag_recall",
                "policy_reading_precision", "policy_reading_recall", "rule_precision", "rule_recall",
                "errors", "judge_flags_rejected", "retries_used", "template_fallbacks")


class EvalGuardError(RuntimeError):
    """The requested evaluation is not allowed (test split without --final, hash mismatch, or a repeat final run)."""


class ConsecutiveFailuresError(RuntimeError):
    """ABORT_AFTER_CONSECUTIVE_FAILURES claims in a row ended llm_failed or exception (D64). failures holds
    (run, claim id, status, sanitized error texts) for each of them, oldest first."""

    def __init__(self, failures: list[tuple[int, str, str, list[str]]]):
        self.failures = failures
        super().__init__(f"{len(failures)} consecutive failed claims (llm_failed or exception); "
                         "aborting with no results file.")


def sanitize_errors(errors) -> list[str]:
    """Error texts safe to store: key-like gsk_ strings replaced by [redacted], each cut to ERROR_TEXT_LIMIT chars."""
    return [KEY_PATTERN.sub("[redacted]", str(e))[:ERROR_TEXT_LIMIT] for e in (errors or [])]


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


# ------------------------------------------------------------------ splits and guards
def split_records(claims: list[dict], gts: list[dict], split: str) -> tuple[list[dict], dict[str, dict]]:
    """Keep only the chosen split: claims in file order and ground truth keyed by claim id."""
    keep = {g["claim_id"]: g for g in gts if g["split"] == split}
    return [c for c in claims if c["claim_id"] in keep], keep


def test_split_hash(claims: list[dict], gts: list[dict]) -> str:
    """SHA256 over the test-split claims and their ground truth, the same payload the generator freezes."""
    ids = {g["claim_id"] for g in gts if g["split"] == "test"}
    payload = {
        "claims": sorted((c for c in claims if c["claim_id"] in ids), key=lambda c: c["claim_id"]),
        "ground_truth": sorted((g for g in gts if g["claim_id"] in ids), key=lambda g: g["claim_id"]),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_guards(split: str, final: bool, out_path: Path, final_out_path: Path,
                 load_records: Callable[[], tuple[list[dict], list[dict]]] | None = None,
                 frozen_hash_path: Path | None = None) -> None:
    """Raise EvalGuardError unless this run is allowed. Reads the frozen hash file, never writes it.

    dev: --final is refused (the final flag exists only for the one test run).
    test: needs --final, the output must be the fixed final path, that file must not exist yet
    (one final run only, never overwritten), and the test-split hash must match the frozen file.
    load_records returns (all claims, all ground truth) and is called only after the cheap checks pass.
    """
    if split not in ("dev", "test"):
        raise EvalGuardError(f"unknown split {split!r}")
    if split == "dev":
        if final:
            raise EvalGuardError("--final is only for the one test-split run.")
        return
    if not final:
        raise EvalGuardError("Refusing to run on the test split without --final (D14).")
    if Path(out_path).resolve() != Path(final_out_path).resolve():
        raise EvalGuardError(f"The final test run must write {final_out_path}, not another path.")
    if Path(final_out_path).exists():
        raise EvalGuardError(f"{final_out_path} already exists. The final test run happens once and is never overwritten.")
    if load_records is None or frozen_hash_path is None:
        raise EvalGuardError("The test-split hash cannot be checked without the records and the frozen hash file.")
    claims, gts = load_records()
    frozen = Path(frozen_hash_path).read_text(encoding="utf-8").strip()
    if test_split_hash(claims, gts) != frozen:
        raise EvalGuardError("The test-split hash does not match the frozen file. Refusing to run.")


# ------------------------------------------------------------------ scoring
def prediction(result: dict) -> str:
    """Flagged, Clear or Error for one agent result dict (anything not a clean ok/flag/pass is an Error)."""
    if result.get("status") != "ok":
        return "Error"
    action = result.get("recommended_action")
    if action == "flag":
        return "Flagged"
    if action == "pass":
        return "Clear"
    return "Error"


def predicted_flags(result: dict) -> set[str] | None:
    """The raised flag ids, or None when the claim is an Error (an error raises nothing)."""
    if prediction(result) == "Error":
        return None
    return {f["flag_id"] for f in result.get("flags", [])}


def flag_scores(expected: dict[str, set[str]], predicted: dict[str, set[str] | None],
                keep: Callable[[str], bool] = lambda _flag: True) -> dict:
    """Micro precision and recall over (claim, flag) pairs for the flag ids that keep() accepts.

    expected and predicted are keyed by claim id; predicted None means the claim failed, so its
    expected flags are all false negatives. A claim missing from predicted is treated the same way.
    """
    tp = fp = fn = 0
    for cid, exp in expected.items():
        exp = {f for f in exp if keep(f)}
        got = predicted.get(cid)
        got = set() if got is None else {f for f in got if keep(f)}
        tp += len(exp & got)
        fp += len(got - exp)
        fn += len(exp - got)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn)}


def _retries(result: dict) -> int:
    """Extra LLM attempts (judge and explanation) taken for one claim, read from the trace."""
    total = 0
    for entry in result.get("trace", []):
        if entry.get("node") in ("judge", "explain"):
            attempts = (entry.get("data") or {}).get("attempts") or []
            total += max(len(attempts) - 1, 0)
    return total


def score_results(results: list[dict], gts: dict[str, dict], detection: dict[str, str]) -> dict:
    """Score one run. results are agent result dicts (one per claim); gts maps claim id to ground truth.

    detection maps flag id to "rule" or "policy_reading" (from the catalog). An unknown flag id
    counts in the overall flag metrics but in neither subset.
    """
    by_id = {r["claim_id"]: r for r in results}
    missing = sorted(set(gts) - set(by_id))
    if missing:
        raise ValueError(f"no result for {len(missing)} claim(s): {missing}")
    rows = [(gts[cid]["is_anomalous"], prediction(by_id[cid])) for cid in gts]
    decisions = compute_metrics(rows)
    expected = {cid: set(g["expected_flags"]) for cid, g in gts.items()}
    predicted = {cid: predicted_flags(by_id[cid]) for cid in gts}
    overall = flag_scores(expected, predicted)
    policy = flag_scores(expected, predicted, lambda f: detection.get(f) == "policy_reading")
    rule = flag_scores(expected, predicted, lambda f: detection.get(f) == "rule")

    per_claim, error_ids = {}, []
    for cid in sorted(gts):
        got = predicted[cid]
        if got is None:
            error_ids.append(cid)
        missed = sorted(expected[cid] - (got or set()))
        false_pos = sorted((got or set()) - expected[cid])
        failed = by_id[cid].get("status") == "llm_failed"
        if missed or false_pos or failed:
            per_claim[cid] = {"missed": missed, "false_positive": false_pos}
        if failed:
            per_claim[cid]["llm_errors"] = sanitize_errors(by_id[cid].get("errors"))

    return {
        "claims": len(gts),
        "decision_accuracy": decisions["accuracy"],
        "decision_confusion": {k: decisions[k] for k in ("tp", "fp", "tn", "fn")},
        "errors": decisions["errors"],
        "error_claim_ids": error_ids,
        "flag_precision": overall["precision"], "flag_recall": overall["recall"],
        "flag_counts": {k: overall[k] for k in ("tp", "fp", "fn")},
        "policy_reading_precision": policy["precision"], "policy_reading_recall": policy["recall"],
        "policy_reading_counts": {k: policy[k] for k in ("tp", "fp", "fn")},
        "rule_precision": rule["precision"], "rule_recall": rule["recall"],
        "rule_counts": {k: rule[k] for k in ("tp", "fp", "fn")},
        "judge_flags_rejected": sum(len(by_id[cid].get("rejected") or []) for cid in gts),
        "retries_used": sum(_retries(by_id[cid]) for cid in gts),
        "template_fallbacks": sum(1 for cid in gts if by_id[cid].get("explanation_source") == "template"),
        "per_claim": per_claim,
    }


def summarize(runs: list[dict]) -> dict:
    """Mean and min to max of every summary metric across runs. None values (undefined) are left out
    of the mean and range, and n_defined says how many runs had a value."""
    out = {}
    for key in SUMMARY_KEYS:
        values = [r[key] for r in runs if r.get(key) is not None]
        out[key] = {"mean": sum(values) / len(values) if values else None,
                    "min": min(values) if values else None,
                    "max": max(values) if values else None,
                    "n_defined": len(values)}
    return out


def baseline_block(claims: list[dict], gts: dict[str, dict], detection: dict[str, str]) -> dict:
    """The rules-only baseline on the same claims, scored with the same functions as the agent."""
    results = []
    for r in run_rules_only(claims):
        if r.decision == "Error":
            results.append({"claim_id": r.claim_id, "status": "error", "recommended_action": None, "flags": []})
        else:
            results.append({"claim_id": r.claim_id, "status": "ok",
                            "recommended_action": "flag" if r.decision == "Flagged" else "pass",
                            "flags": [{"flag_id": f.flag_id} for f in r.flags]})
    scored = score_results(results, gts, detection)
    for key in ("judge_flags_rejected", "retries_used", "template_fallbacks"):
        scored.pop(key)
    return {"label": BASELINE_LABEL, **scored}


# ------------------------------------------------------------------ running
def run_split(claims: list[dict], review: Callable[[dict], dict], runs: int,
              progress: Callable[[int, str, str], None] = lambda run, cid, status: None) -> list[list[dict]]:
    """Run review() on every claim, runs times, sequentially. An exception becomes an error result
    (status "exception"), never a dropped claim (D13). progress gets (run number, claim id, status).

    Failure guard (D64): if ABORT_AFTER_CONSECUTIVE_FAILURES claims in a row end llm_failed or exception
    (STREAK_STATUSES; mixed statuses count toward the same streak), counted across runs,
    ConsecutiveFailuresError is raised and nothing is returned. Any other status resets the count."""
    if runs < 1:
        raise ValueError("runs must be at least 1")
    all_runs, streak = [], []
    for run in range(1, runs + 1):
        results = []
        for claim in claims:
            try:
                result = review(claim)
            except Exception as exc:
                result = {"claim_id": claim.get("claim_id"), "status": "exception", "recommended_action": None,
                          "flags": [], "rejected": [], "trace": [], "explanation_source": None,
                          "errors": [f"{type(exc).__name__}: {exc}"]}
            results.append(result)
            progress(run, result["claim_id"], result["status"])
            if result.get("status") in STREAK_STATUSES:
                streak.append((run, result["claim_id"], result["status"], sanitize_errors(result.get("errors"))))
                if len(streak) >= ABORT_AFTER_CONSECUTIVE_FAILURES:
                    raise ConsecutiveFailuresError(streak[-ABORT_AFTER_CONSECUTIVE_FAILURES:])
            else:
                streak = []
        all_runs.append(results)
    return all_runs


def build_output(*, split: str, model: str, temperature, runs_results: list[list[dict]], gts: dict[str, dict],
                 claims: list[dict], detection: dict[str, str], commit: str, now: datetime | None = None) -> dict:
    """The full results file content."""
    per_run = [score_results(results, gts, detection) for results in runs_results]
    now = now or datetime.now(timezone.utc)
    return {
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "claim_count": len(gts),
        "runs": len(per_run),
        "model": model,
        "temperature": temperature,
        "date_utc": now.astimezone(timezone.utc).replace(microsecond=0).isoformat(),
        "git_commit": commit,
        "agent": {"per_run": per_run, "summary": summarize(per_run)},
        "rules_only_baseline": baseline_block(claims, gts, detection),
    }
