"""
Run the rules-only baseline on one split and print metrics and per-flag counts.

Usage:
    .venv/bin/python scripts/run_baseline.py --split dev [--show-misses]
    .venv/bin/python scripts/run_baseline.py --split test --final    (Phase 6 only, see D14)

Only the chosen split's claims are used, including for the cross-claim checks.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from claims_review.baseline import run_rules_only  # noqa: E402
from claims_review.metrics import compute_metrics, format_ratio, per_flag_counts  # noqa: E402

V2 = ROOT / "data" / "v2"


def fmt(value):
    return "n/a (undefined)" if value is None else f"{value:.3f}"


def main():
    ap = argparse.ArgumentParser(description="Rules-only baseline on a dataset v2 split.")
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--show-misses", action="store_true", help="list claims the baseline got wrong")
    ap.add_argument("--final", action="store_true", help="required to touch the frozen test split")
    args = ap.parse_args()

    if args.split == "test" and not args.final:
        print("Refusing to run on the test split without --final (tuning protocol D14: test runs only in Phase 6).",
              file=sys.stderr)
        return 2

    claims = json.loads((V2 / "claims.json").read_text(encoding="utf-8"))
    gts = {g["claim_id"]: g for g in json.loads((V2 / "ground_truth.json").read_text(encoding="utf-8"))}
    claims = [c for c in claims if gts[c["claim_id"]]["split"] == args.split]
    results = run_rules_only(claims)

    rows = [(gts[r.claim_id]["is_anomalous"], r.decision) for r in results]
    m = compute_metrics(rows)
    n_anom = sum(1 for a, _ in rows if a)
    print(f"RULES-ONLY BASELINE, split={args.split}: {m['n']} claims ({n_anom} anomalous, {m['n'] - n_anom} not)")
    print(f"  decisions: flagged={sum(r.decision == 'Flagged' for r in results)} "
          f"clear={sum(r.decision == 'Clear' for r in results)} error={m['errors']}")
    print(f"  confusion (errors counted as wrong): TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}")
    print(f"  accuracy={fmt(m['accuracy'])} precision={fmt(m['precision'])} recall={fmt(m['recall'])} "
          f"fpr={fmt(m['fpr'])} f1={fmt(m['f1'])}")
    print(f"  errors: {m['errors']} total ({m['errors_on_anomalous']} on anomalous, {m['errors_on_clean']} on clean)")

    expected = {c: set(g["expected_flags"]) for c, g in gts.items() if g["split"] == args.split}
    predicted = {r.claim_id: (None if r.decision == "Error" else {f.flag_id for f in r.flags}) for r in results}
    print("\nPER FLAG (hit = the flag was raised on claims that should raise it)")
    print("  " + "flag".ljust(28) + "hits".rjust(10) + "  false alarms")
    for flag, c in per_flag_counts(expected, predicted).items():
        print("  " + flag.ljust(28) + format_ratio(c["hit"], c["total"]).rjust(10) + f"  {c['false_alarm']}")

    if args.show_misses:
        print("\nMISSES (claims where the decision is wrong)")
        shown = 0
        for r in results:
            g = gts[r.claim_id]
            wrong = r.decision == "Error" or (r.decision == "Flagged") != g["is_anomalous"]
            if wrong:
                shown += 1
                got = "error: " + (r.error or "") if r.decision == "Error" else ", ".join(f.flag_id for f in r.flags) or "none"
                print(f"  {r.claim_id} [{g['category']}] decision={r.decision} expected={g['expected_flags'] or 'none'} "
                      f"raised={got}")
                print(f"      why: {g['notes']}")
        if not shown:
            print("  (none)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
