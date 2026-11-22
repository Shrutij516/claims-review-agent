"""
Measure BM25 retrieval of the gold policy evidence on a dataset v2 split. Three variants:

  (a) criterion level, top 3 criteria, plain query            (the Phase 2 retriever)
  (b) whole policy, top 2 policies, plain query
  (c) whole policy, top 2 policies, flag-aware query          (rule flags appended to the query)

Only claims whose ground truth cites at least one policy criterion are used. The plain query is the
procedure label, the diagnosis labels and the clinical note. Flags for (c) come from rule_flags, using
only the chosen split's claims as the store (D19).

Selection rule, fixed before these numbers were seen (D21): use (c) only if its policy recall@2 is
strictly higher than that of (b); otherwise use (b). No further variants and no further tuning.

Usage:
    .venv/bin/python scripts/eval_retrieval.py --split dev
    .venv/bin/python scripts/eval_retrieval.py --split test --final    (Phase 6 only, see D14)

Metrics are also saved to results/retrieval_<split>.json.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from claims_review.codes import load_tables  # noqa: E402
from claims_review.parsing import parse_claim  # noqa: E402
from claims_review.retrieval import make_query, retrieve, retrieve_policies  # noqa: E402
from claims_review.rules import rule_flags  # noqa: E402

V2 = ROOT / "data" / "v2"
RESULTS = ROOT / "results"
K_CRITERIA = 3
K_POLICIES = 2


def policy_of(criterion_id: str) -> str:
    return criterion_id.split(".")[0]


def ratio(found: int, total: int) -> float | None:
    return found / total if total else None


def fmt(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Retrieval recall of gold policy evidence on a dataset v2 split.")
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--final", action="store_true", help="required to touch the frozen test split")
    args = ap.parse_args()

    if args.split == "test" and not args.final:
        print("Refusing to run on the test split without --final (tuning protocol D14: test runs only in Phase 6).",
              file=sys.stderr)
        return 2

    tables = load_tables()
    raws = json.loads((V2 / "claims.json").read_text(encoding="utf-8"))
    gts = {g["claim_id"]: g for g in json.loads((V2 / "ground_truth.json").read_text(encoding="utf-8"))}
    split_claims = [parse_claim(r) for r in raws if gts[r["claim_id"]]["split"] == args.split]

    # per-variant tallies
    a = {"gold": 0, "found": 0, "complete": 0}
    pol = {v: {"gold_policies": 0, "found_policies": 0, "policies_complete": 0,
               "gold_criteria": 0, "covered_criteria": 0, "criteria_complete": 0} for v in ("b", "c")}
    misses = {"a": [], "b": [], "c": []}
    evaluated = 0

    for claim in split_claims:
        gold = {e["id"] for e in gts[claim.claim_id]["evidence"] if e["kind"] == "policy"}
        if not gold:
            continue
        evaluated += 1
        gold_pols = {policy_of(c) for c in gold}

        plain = make_query(claim, tables=tables)
        got_a = {h.criterion_id for h in retrieve(plain, k=K_CRITERIA)}
        a["gold"] += len(gold)
        a["found"] += len(gold & got_a)
        a["complete"] += gold <= got_a
        if not gold <= got_a:
            misses["a"].append((claim.claim_id, sorted(gold - got_a)))

        flags = rule_flags(claim, split_claims, tables)
        for v, query in (("b", plain), ("c", make_query(claim, flags=flags, tables=tables))):
            hits = retrieve_policies(query, k=K_POLICIES)
            got_pols = {h.policy_id for h in hits}
            got_crit = {cid for h in hits for cid, _ in h.criteria}
            t = pol[v]
            t["gold_policies"] += len(gold_pols)
            t["found_policies"] += len(gold_pols & got_pols)
            t["policies_complete"] += gold_pols <= got_pols
            t["gold_criteria"] += len(gold)
            t["covered_criteria"] += len(gold & got_crit)
            t["criteria_complete"] += gold <= got_crit
            if not gold_pols <= got_pols:
                misses[v].append((claim.claim_id, sorted(gold_pols - got_pols), sorted(got_pols)))

    print(f"BM25 RETRIEVAL, split={args.split}")
    print(f"  claims with policy evidence: {evaluated}")
    if not evaluated:
        print("  nothing to evaluate")
        return 0

    print(f"  (a) criterion level, top {K_CRITERIA}, plain query")
    print(f"      criterion recall@{K_CRITERIA}: {a['found']} of {a['gold']} gold criteria ({fmt(ratio(a['found'], a['gold']))}); "
          f"claims with all gold criteria: {a['complete']} of {evaluated}")
    results = {"split": args.split, "claims_evaluated": evaluated,
               "a_criterion_top3_plain": {**a, "recall": ratio(a["found"], a["gold"])}}
    for v, name in (("b", "plain query"), ("c", "flag-aware query")):
        t = pol[v]
        rp = ratio(t["found_policies"], t["gold_policies"])
        cc = ratio(t["covered_criteria"], t["gold_criteria"])
        print(f"  ({v}) whole policy, top {K_POLICIES}, {name}")
        print(f"      policy recall@{K_POLICIES}: {t['found_policies']} of {t['gold_policies']} gold policies ({fmt(rp)}); "
              f"claims with all gold policies: {t['policies_complete']} of {evaluated}")
        print(f"      criterion coverage: {t['covered_criteria']} of {t['gold_criteria']} gold criteria inside the returned "
              f"policies ({fmt(cc)}); claims fully covered: {t['criteria_complete']} of {evaluated}")
        results[f"{v}_policy_top2_{'plain' if v == 'b' else 'flag_aware'}"] = {
            **t, "policy_recall": rp, "criterion_coverage": cc}

    rp_b = ratio(pol["b"]["found_policies"], pol["b"]["gold_policies"])
    rp_c = ratio(pol["c"]["found_policies"], pol["c"]["gold_policies"])
    selected = "c" if (rp_c is not None and rp_b is not None and rp_c > rp_b) else "b"
    print("  SELECTION RULE (D21): use (c) only if its policy recall@2 is strictly higher than (b)")
    print(f"      (b) {fmt(rp_b)} vs (c) {fmt(rp_c)} -> selected ({selected})")
    results["selection_rule"] = "use (c) only if policy recall@2 is strictly higher than (b), else (b)"
    results["selected_variant"] = selected

    print("  claims with a missing gold item:")
    for v, label in (("a", "(a) missing criteria"), ("b", "(b) missing policies"), ("c", "(c) missing policies")):
        print(f"    {label}:")
        for item in misses[v]:
            print(f"      {item[0]}: {', '.join(item[1])}" + (f"; retrieved {', '.join(item[2])}" if len(item) > 2 else ""))
        if not misses[v]:
            print("      (none)")
    results["misses"] = {v: [list(m) for m in ms] for v, ms in misses.items()}

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"retrieval_{args.split}.json"
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"  saved {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
