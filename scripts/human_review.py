"""
Human review of one claim: start (agent runs, then pauses), show, submit the reviewer's action.

`start` makes live LLM calls (Groq) and needs GROQ_API_KEY in the environment. `show` and `submit`
make no LLM call. Pauses are stored in SQLite (default runs/checkpoints.sqlite, gitignored); the
thread id is the claim id, so there is one review per claim.

Usage:
    .venv/bin/python scripts/human_review.py start  --claim V2-0001 --split dev
    .venv/bin/python scripts/human_review.py show   --claim V2-0001 --split dev
    .venv/bin/python scripts/human_review.py submit --claim V2-0001 --split dev --action approve --reviewer alex
    .venv/bin/python scripts/human_review.py submit --claim V2-0001 --split dev --action override \\
        --decision pass --reason "Note documents 8 weeks of therapy"
    .venv/bin/python scripts/human_review.py submit --claim V2-0001 --split dev --action request_info \\
        --question "Which knee was imaged?"

The test split is refused without --final (Phase 6 only, see D14).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from claims_review.graph import build_store  # noqa: E402
from claims_review.llm import GroqLLM, MissingAPIKeyError  # noqa: E402
from claims_review.review import (ReviewActionError, ReviewExistsError, close_saver, get_review,  # noqa: E402
                                  open_saver, start_review, submit_review)

V2 = ROOT / "data" / "v2"
DEFAULT_DB = "runs/checkpoints.sqlite"


def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--claim", required=True, help="claim id, for example V2-0001")
    common.add_argument("--split", choices=["dev", "test"], required=True)
    common.add_argument("--final", action="store_true", help="required to touch the frozen test split")
    common.add_argument("--db", default=DEFAULT_DB, help=f"SQLite checkpoint file (default {DEFAULT_DB})")

    ap = argparse.ArgumentParser(description="Human review of one claim.")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("start", parents=[common], help="run the agent and pause for review")
    sub.add_parser("show", parents=[common], help="show the stored review state")
    sp = sub.add_parser("submit", parents=[common], help="submit the reviewer's action")
    sp.add_argument("--action", required=True, help="approve, override or request_info")
    sp.add_argument("--reviewer", default="reviewer")
    sp.add_argument("--decision", help="for override: pass or flag")
    sp.add_argument("--reason", help="for override")
    sp.add_argument("--question", help="for request_info")
    return ap


def main():
    args = build_parser().parse_args()

    if args.split == "test" and not args.final:
        print("Refusing to run on the test split without --final (tuning protocol D14: test runs only in Phase 6).",
              file=sys.stderr)
        return 2

    db = Path(args.db)
    db = db if db.is_absolute() else ROOT / db

    if args.command == "start":
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
        saver = open_saver(db)
        try:
            out = start_review(chosen[0], llm, build_store(claims), saver)
        except ReviewExistsError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        finally:
            close_saver(saver)
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0

    saver = open_saver(db)
    try:
        if args.command == "show":
            print(json.dumps(get_review(args.claim, saver), indent=2, ensure_ascii=False))
            return 0
        action = {"action": args.action, "reviewer": args.reviewer}
        for key, value in (("final_decision", args.decision), ("reason", args.reason), ("question", args.question)):
            if value is not None:
                action[key] = value
        try:
            record = submit_review(args.claim, action, saver)
        except ReviewActionError as exc:
            print("The action was not accepted and nothing was resumed:", file=sys.stderr)
            for problem in exc.problems:
                print(f"  - {problem}", file=sys.stderr)
            return 2
        print(json.dumps(record, indent=2, ensure_ascii=False))
        return 0
    finally:
        close_saver(saver)


if __name__ == "__main__":
    sys.exit(main())
