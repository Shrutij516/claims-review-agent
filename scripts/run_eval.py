"""
Evaluate the plain agent (run_claim, no human review, D37) on one split, several runs, and save the results.

Usage:
    .venv/bin/python scripts/run_eval.py                        (dev split, 3 runs, results/eval_dev.json)
    .venv/bin/python scripts/run_eval.py --split dev --runs 3 --out results/eval_dev.json
    .venv/bin/python scripts/run_eval.py --split test --final   (Phase 6 final run only, once, see D57)

Claims are reviewed sequentially with GroqLLM (GROQ_API_KEY from the environment). Only the chosen
split's claims are used, including as the cross-claim store. One progress line is printed per claim
(run, claim id, status) and nothing else about claims. The rules-only baseline on the same split is
saved in the same file.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from claims_review.evaluation import (EvalGuardError, build_output, check_guards, run_split,  # noqa: E402
                                      split_records)
from claims_review.graph import build_store, run_claim  # noqa: E402
from claims_review.llm import GroqLLM, MissingAPIKeyError  # noqa: E402
from claims_review.rules import load_catalog  # noqa: E402

V2 = ROOT / "data" / "v2"
RESULTS = ROOT / "results"


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return sha + ("+dirty" if dirty else "")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate the review agent on a dataset v2 split.")
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None, help="default results/eval_<split>.json")
    ap.add_argument("--final", action="store_true", help="required for the one final test-split run")
    args = ap.parse_args(argv)
    if args.runs < 1:
        ap.error("--runs must be at least 1")
    if args.out is None:
        args.out = RESULTS / f"eval_{args.split}.json"
    elif not args.out.is_absolute():
        args.out = ROOT / args.out
    return args


def main(argv=None, data_dir: Path = V2, results_dir: Path = RESULTS, llm_factory=GroqLLM) -> int:
    args = parse_args(argv)
    final_out = results_dir / "eval_test.json"
    loaded = {}

    def load_records():
        if not loaded:
            loaded["claims"] = json.loads((data_dir / "claims.json").read_text(encoding="utf-8"))
            loaded["gts"] = json.loads((data_dir / "ground_truth.json").read_text(encoding="utf-8"))
        return loaded["claims"], loaded["gts"]

    try:
        check_guards(args.split, args.final, args.out, final_out, load_records, data_dir / "TEST_SPLIT.sha256")
    except EvalGuardError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    all_claims, all_gts = load_records()
    claims, gts = split_records(all_claims, all_gts, args.split)
    del all_claims, all_gts
    try:
        llm = llm_factory()
    except MissingAPIKeyError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    store = build_store(claims)
    detection = {f["flag_id"]: f["detection"] for f in load_catalog()["flags"]}
    print(f"eval split={args.split} claims={len(claims)} runs={args.runs} model={getattr(llm, 'model', 'unknown')}")
    runs_results = run_split(claims, lambda c: run_claim(c, llm, store), args.runs,
                             progress=lambda run, cid, status: print(f"run {run} {cid} {status}", flush=True))
    out = build_output(split=args.split, model=getattr(llm, "model", "unknown"),
                       temperature=getattr(llm, "temperature", None), runs_results=runs_results, gts=gts,
                       claims=claims, detection=detection, commit=git_commit())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(out, indent=2, ensure_ascii=False) + "\n"
    # The final test file is created exclusively, so it can never overwrite an earlier final run.
    with open(args.out, "x" if args.split == "test" else "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
