"""
Consistency check, not a performance measure: do the evidence ids that rule_flags cites match the
evidence ids written in the ground truth?

Uses all 60 claims and the full store. Covers every claim whose expected_flags contain an R_ flag.
Report only: this script changes nothing and always exits 0.

Usage:
    .venv/bin/python scripts/check_evidence.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from claims_review.parsing import parse_claim  # noqa: E402
from claims_review.rules import rule_flags  # noqa: E402

V2 = ROOT / "data" / "v2"


def policy_of(criterion_id: str) -> str:
    """POL-003.2 -> POL-003"""
    return criterion_id.split(".")[0]


def show(ids) -> str:
    return ", ".join(sorted(ids)) if ids else "(none)"


def main() -> int:
    raws = json.loads((V2 / "claims.json").read_text(encoding="utf-8"))
    gts = {g["claim_id"]: g for g in json.loads((V2 / "ground_truth.json").read_text(encoding="utf-8"))}
    claims = [parse_claim(r) for r in raws]

    print("EVIDENCE AGREEMENT (rule_flags evidence vs ground-truth evidence), all 60 claims, full store")
    print("  policy-level: policy ids only; criterion-level: full criterion ids like POL-003.2")
    print()

    checked = policy_mismatch = criterion_mismatch = 0
    lines_out, mismatches = [], []
    for claim in claims:
        gt = gts[claim.claim_id]
        if not any(f.startswith("R_") for f in gt["expected_flags"]):
            continue
        rule_crit = {e["id"] for f in rule_flags(claim, claims) for e in f.evidence if e["kind"] == "policy"}
        gt_crit = {e["id"] for e in gt["evidence"] if e["kind"] == "policy"}
        rule_pol = {policy_of(c) for c in rule_crit}
        gt_pol = {policy_of(c) for c in gt_crit}
        checked += 1
        pol_ok, crit_ok = rule_pol == gt_pol, rule_crit == gt_crit
        policy_mismatch += not pol_ok
        criterion_mismatch += not crit_ok
        flags = ", ".join(f for f in gt["expected_flags"] if f.startswith("R_"))
        lines_out.append(f"  {claim.claim_id}  [{flags}]\n"
                         f"      rules:  policies {show(rule_pol)} | criteria {show(rule_crit)}\n"
                         f"      truth:  policies {show(gt_pol)} | criteria {show(gt_crit)}")
        if not (pol_ok and crit_ok):
            mismatches.append((claim.claim_id, pol_ok, crit_ok, rule_pol, gt_pol, rule_crit, gt_crit))

    print("\n".join(lines_out))
    print()
    print(f"claims checked (expected_flags contain an R_ flag): {checked}")
    print(f"policy-level mismatches:    {policy_mismatch} of {checked}")
    print(f"criterion-level mismatches: {criterion_mismatch} of {checked}")
    print("MISMATCHES:")
    if not mismatches:
        print("  (none)")
    for cid, pol_ok, crit_ok, rp, gp, rc, gc in mismatches:
        level = "policy+criterion" if not pol_ok else "criterion only"
        print(f"  {cid} ({level}): rules-only {show(rc - gc)}; truth-only {show(gc - rc)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
