"""
Measure BM25 retrieval: recall@3 of the gold policy criterion ids.

Only claims whose ground truth cites at least one policy criterion are used.
The query is the procedure label, the diagnosis labels and the clinical note.

Usage:
    .venv/bin/python scripts/eval_retrieval.py --split dev
    .venv/bin/python scripts/eval_retrieval.py --split test --final    (Phase 6 only, see D14)
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from claims_review.codes import load_tables  # noqa: E402
from claims_review.parsing import parse_claim  # noqa: E402
from claims_review.retrieval import retrieve  # noqa: E402

V2 = ROOT / "data" / "v2"
K = 3


def build_query(raw: dict, tables) -> str:
    claim = parse_claim(raw)
    proc = tables.cpt.get(claim.cpt) or tables.hcpcs.get(claim.cpt) or {}
    labels = [proc.get("short_desc", "")]
    labels += [tables.icd10[i]["short_desc"] for i in claim.icd10 if i in tables.icd10]
    return " ".join(labels + [claim.clinical_note])


def main():
    ap = argparse.ArgumentParser(description="Recall@3 of gold policy criteria on a dataset v2 split.")
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--final", action="store_true", help="required to touch the frozen test split")
    args = ap.parse_args()

    if args.split == "test" and not args.final:
        print("Refusing to run on the test split without --final (tuning protocol D14: test runs only in Phase 6).",
              file=sys.stderr)
        return 2

    tables = load_tables()
    claims = json.loads((V2 / "claims.json").read_text(encoding="utf-8"))
    gts = {g["claim_id"]: g for g in json.loads((V2 / "ground_truth.json").read_text(encoding="utf-8"))}

    evaluated = gold_total = found_total = full = 0
    recalls, misses = [], []
    for raw in claims:
        g = gts[raw["claim_id"]]
        gold = {e["id"] for e in g["evidence"] if e["kind"] == "policy"}
        if g["split"] != args.split or not gold:
            continue
        got = {h.criterion_id for h in retrieve(build_query(raw, tables), k=K)}
        found = gold & got
        evaluated += 1
        gold_total += len(gold)
        found_total += len(found)
        full += gold <= got
        recalls.append(len(found) / len(gold))
        if found != gold:
            misses.append((raw["claim_id"], sorted(gold - got), sorted(got)))

    print(f"BM25 RETRIEVAL, split={args.split}, k={K}")
    print(f"  claims with policy evidence: {evaluated}")
    if not evaluated:
        print("  nothing to evaluate")
        return 0
    print(f"  gold criteria found in top {K}: {found_total} of {gold_total} (micro recall@{K} = {found_total / gold_total:.3f})")
    print(f"  mean per-claim recall@{K} = {sum(recalls) / len(recalls):.3f}")
    print(f"  claims with every gold criterion in top {K}: {full} of {evaluated}")
    print("  claims with a missing gold criterion:")
    for cid, missing, got in misses:
        print(f"    {cid}: missing {missing}; retrieved {got}")
    if not misses:
        print("    (none)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
