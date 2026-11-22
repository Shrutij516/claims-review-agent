"""
Print counts for dataset v2. With --examples, also print one clean claim, one
hard negative, one rule anomaly and one policy reading anomaly, each with its
ground truth and the linked policy text.

Usage:
    .venv/bin/python scripts/dataset_stats.py
    .venv/bin/python scripts/dataset_stats.py --examples
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import generate_claims_v2 as gen  # noqa: E402

V2 = ROOT / "data" / "v2"


def load():
    claims = json.loads((V2 / "claims.json").read_text(encoding="utf-8"))
    gts = json.loads((V2 / "ground_truth.json").read_text(encoding="utf-8"))
    return claims, gts


def print_counts(claims, gts):
    splits = ("dev", "test")
    print("COUNTS")
    print(f"  total claims: {len(claims)}")
    for s in splits:
        part = [g for g in gts if g["split"] == s]
        anom = sum(g["is_anomalous"] for g in part)
        print(f"  {s}: {len(part)} claims, {anom} anomalous ({anom / len(part):.1%})")

    print("\nBY SPLIT AND CATEGORY")
    cats = ("clean", "hard_negative", "anomaly")
    print("  " + "split".ljust(6) + "".join(c.rjust(15) for c in cats))
    for s in splits:
        row = Counter(g["category"] for g in gts if g["split"] == s)
        print("  " + s.ljust(6) + "".join(str(row[c]).rjust(15) for c in cats))

    print("\nBY ANOMALY TYPE (expected flag; a claim with several flags counts once per flag)")
    flags = sorted({f for g in gts for f in g["expected_flags"]})
    print("  " + "flag".ljust(28) + "dev".rjust(6) + "test".rjust(6) + "  detection")
    catalog = {f["flag_id"]: f["detection"] for f in gen.load_catalog()["flags"]}
    for f in flags:
        d = sum(f in g["expected_flags"] for g in gts if g["split"] == "dev")
        t = sum(f in g["expected_flags"] for g in gts if g["split"] == "test")
        print("  " + f.ljust(28) + str(d).rjust(6) + str(t).rjust(6) + "  " + catalog[f])

    print("\nPOLICY READING")
    for s in splits:
        part = [g for g in gts if g["split"] == s]
        reading = [g for g in part if g["needs_policy_reading"]]
        anom_reading = [g for g in reading if g["is_anomalous"]]
        anom = [g for g in part if g["is_anomalous"]]
        print(f"  {s}: {len(reading)} claims need policy reading; "
              f"{len(anom_reading)} of {len(anom)} anomalies ({len(anom_reading) / len(anom):.1%})")

    frozen_path = V2 / "TEST_SPLIT.sha256"
    frozen = frozen_path.read_text(encoding="utf-8").strip() if frozen_path.exists() else "(missing)"
    current = gen.test_split_hash(claims, gts)
    print("\nTEST SPLIT HASH")
    print(f"  computed: {current}")
    print(f"  frozen:   {frozen}")
    print(f"  match:    {current == frozen}")


def pick_examples(claims, gts):
    by_id = {c["claim_id"]: c for c in claims}
    chosen = []

    def first(label, predicate):
        for g in gts:
            if predicate(g):
                chosen.append((label, by_id[g["claim_id"]], g))
                return

    def rule_only(g):
        return g["is_anomalous"] and all(f.startswith("R_") for f in g["expected_flags"])

    first("CLEAN", lambda g: g["category"] == "clean" and g["evidence"])
    first("HARD NEGATIVE", lambda g: g["category"] == "hard_negative")
    first("RULE ANOMALY", lambda g: rule_only(g) and any(e["kind"] == "policy" for e in g["evidence"]))
    first("POLICY READING ANOMALY", lambda g: g["is_anomalous"] and g["needs_policy_reading"])
    return chosen


def print_examples(claims, gts):
    criteria = gen.load_policy_criteria()
    for label, claim, gt in pick_examples(claims, gts):
        print("\n" + "=" * 72)
        print(label)
        print("=" * 72)
        print("CLAIM")
        print(json.dumps(claim, indent=2, ensure_ascii=False))
        print("GROUND TRUTH")
        print(json.dumps(gt, indent=2, ensure_ascii=False))
        print("LINKED POLICY TEXT")
        linked = [e["id"] for e in gt["evidence"] if e["kind"] == "policy"]
        if not linked:
            print("  (no policy criterion linked)")
        for pid in linked:
            print(f"  {pid}: {criteria[pid]}")


def main():
    ap = argparse.ArgumentParser(description="Dataset v2 statistics.")
    ap.add_argument("--examples", action="store_true", help="also print four example claims")
    args = ap.parse_args()
    claims, gts = load()
    print_counts(claims, gts)
    if args.examples:
        print_examples(claims, gts)


if __name__ == "__main__":
    main()
