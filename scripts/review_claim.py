"""
Review one claim with the LangGraph agent and print the full result JSON, trace included.

This is the only entry point that makes live LLM calls (Groq). It needs GROQ_API_KEY in the environment.

Usage:
    .venv/bin/python scripts/review_claim.py --claim V2-0001 --split dev
    .venv/bin/python scripts/review_claim.py --claim V2-0001 --split test --final    (Phase 6 only, see D14)

Only the chosen split's claims are loaded, including for the cross-claim rules.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from claims_review.graph import build_store, run_claim  # noqa: E402
from claims_review.llm import GroqLLM, MissingAPIKeyError  # noqa: E402

V2 = ROOT / "data" / "v2"


def main():
    ap = argparse.ArgumentParser(description="Review one claim with the agent (live LLM calls).")
    ap.add_argument("--claim", required=True, help="claim id, for example V2-0001")
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--final", action="store_true", help="required to touch the frozen test split")
    args = ap.parse_args()

    if args.split == "test" and not args.final:
        print("Refusing to run on the test split without --final (tuning protocol D14: test runs only in Phase 6).",
              file=sys.stderr)
        return 2

    claims = json.loads((V2 / "claims.json").read_text(encoding="utf-8"))
    gts = {g["claim_id"]: g for g in json.loads((V2 / "ground_truth.json").read_text(encoding="utf-8"))}
    claims = [c for c in claims if gts[c["claim_id"]]["split"] == args.split]
    chosen = [c for c in claims if c["claim_id"] == args.claim]
    if not chosen:
        print(f"Claim {args.claim} is not in the {args.split} split.", file=sys.stderr)
        return 2

    try:
        llm = GroqLLM()
    except MissingAPIKeyError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    result = run_claim(chosen[0], llm, build_store(claims))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
