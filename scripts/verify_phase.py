"""
End-of-phase verification (D61). Stdlib only, deterministic, no network, no LLM.

Usage:
    .venv/bin/python scripts/verify_phase.py [--base REF] [--after-final]      (default base origin/main)

--after-final is for phases after the one final test run: results/eval_test.json must then exist,
be valid JSON and have split "test". Every other file under results/ must still hold no test-split id.

Prints one line per check, "PASS name: detail" or "FAIL name: detail", and exits 0 only if every
check passes. It prints ids, counts and file names only, never claim or ground-truth text.

The employer-name fragment is assembled at run time, so no tracked file (this one included) contains
it literally; otherwise the names check would always find itself.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FROZEN_HASH = "879e31a49c432d984ffa944dba59a2d3e78d74b18d4dadd2a1350feee0243691"
PROTECTED = ("data/", "docs/policies/", "src/generate_claims_v2.py", "data/v2/TEST_SPLIT.sha256")
NAME_FRAGMENT = "o" + "tiviti"
MAX_UNTRACKED, MAX_SCOPE = 10, 30


def git(root: Path, *args: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=check)


def lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


def line(ok: bool, name: str, detail: str) -> str:
    return f"{'PASS' if ok else 'FAIL'} {name}: {detail}"


# ------------------------------------------------------------------ pure helpers (unit tested)
def is_protected(path: str) -> bool:
    return any(path == p or (p.endswith("/") and path.startswith(p)) for p in PROTECTED)


def numbering(text: str) -> tuple[bool, str]:
    """D and I ids must run 1..n with no gaps or duplicates."""
    out, ok = [], True
    for kind, pattern in (("D", r"^- \*\*D(\d+)\.\*\*"), ("I", r"^- \*\*I(\d+)\*\*")):
        ids = [int(n) for n in re.findall(pattern, text, flags=re.M)]
        good = ids == list(range(1, len(ids) + 1))
        ok &= good and bool(ids)
        out.append(f"{kind} max {max(ids) if ids else 'none'}{'' if good else ' (not consecutive from 1)'}")
    return ok, ", ".join(out)


def count_ci(fragment: str, text: str) -> int:
    return len(re.findall(re.escape(fragment), text, flags=re.I))


def files_with_ids(paths: list[Path], ids: set[str]) -> int:
    """How many files contain at least one of the ids (an id followed by another digit does not count)."""
    if not ids:
        return 0
    pattern = re.compile("(?:" + "|".join(re.escape(i) for i in sorted(ids)) + r")(?!\d)")
    return sum(1 for p in paths if pattern.search(p.read_text(encoding="utf-8", errors="replace")))


def results_check(results_dir: Path) -> tuple[bool, str]:
    files = sorted(results_dir.glob("eval_*.json"))
    if not files:
        return True, "no eval_*.json files"
    ok, parts = True, []
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            ok = False
            parts.append(f"{f.name} unreadable ({type(exc).__name__})")
            continue
        commit = str(d.get("git_commit"))
        ok &= "+dirty" not in commit
        parts.append(f"{f.name} split={d.get('split')} runs={d.get('runs')} "
                     f"claim_count={d.get('claim_count')} git_commit={commit}")
    return ok, "; ".join(parts)


# ------------------------------------------------------------------ checks
def check_head(root: Path) -> str:
    sha = git(root, "rev-parse", "--short", "HEAD").stdout.strip()
    branch = git(root, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    counts = git(root, "rev-list", "--left-right", "--count", "origin/main...HEAD")
    if counts.returncode != 0:
        return line(False, "head", f"{sha} on {branch}; origin/main not found")
    behind, ahead = counts.stdout.split()
    return line(bool(sha), "head", f"{sha} on {branch}, ahead {ahead}, behind {behind} vs origin/main")


def check_tree(root: Path) -> str:
    status = lines(git(root, "status", "--porcelain", "--untracked-files=all").stdout)
    tracked = [s[3:] for s in status if not s.startswith("??")]
    untracked = [s[3:] for s in status if s.startswith("??")]
    shown = ", ".join(untracked[:MAX_UNTRACKED]) + (" ..." if len(untracked) > MAX_UNTRACKED else "")
    detail = f"{len(tracked)} tracked change(s){': ' + ', '.join(tracked[:MAX_UNTRACKED]) if tracked else ''}; " \
             f"{len(untracked)} untracked{': ' + shown if untracked else ''}"
    return line(not status, "tree", detail)


def check_tests(root: Path) -> str:
    proc = subprocess.run([str(root / ".venv" / "bin" / "python"), "-m", "pytest", "-q"], cwd=root,
                          capture_output=True, text=True)
    out = lines(proc.stdout) or lines(proc.stderr) or ["(no output)"]
    return line(proc.returncode == 0, "tests", out[-1].strip())


def changed_files(root: Path, base: str) -> list[str]:
    """Files changed between base and the working tree, plus untracked files."""
    diff = lines(git(root, "diff", "--name-only", base).stdout)
    untracked = lines(git(root, "ls-files", "--others", "--exclude-standard").stdout)
    return sorted(set(diff) | set(untracked))


def check_protected(root: Path, base: str) -> str:
    if git(root, "rev-parse", "--verify", "--quiet", base).returncode != 0:
        return line(False, "protected", f"base {base} not found")
    hits = [f for f in changed_files(root, base) if is_protected(f)]
    return line(not hits, "protected", f"{len(hits)} protected file(s) changed since {base}"
                + (": " + ", ".join(hits) if hits else ""))


def check_frozen(root: Path) -> str:
    sys.path.insert(0, str(root / "src"))
    from claims_review.evaluation import test_split_hash
    v2 = root / "data" / "v2"
    file_ok = (v2 / "TEST_SPLIT.sha256").read_text(encoding="utf-8").strip() == FROZEN_HASH
    claims = json.loads((v2 / "claims.json").read_text(encoding="utf-8"))
    gts = json.loads((v2 / "ground_truth.json").read_text(encoding="utf-8"))
    computed_ok = test_split_hash(claims, gts) == FROZEN_HASH
    return line(file_ok and computed_ok, "frozen", f"file={file_ok} computed={computed_ok}")


def check_names(root: Path) -> str:
    grep = git(root, "grep", "-i", "-c", "-I", "-F", NAME_FRAGMENT)
    in_files = sum(int(x.rsplit(":", 1)[1]) for x in lines(grep.stdout))
    in_messages = count_ci(NAME_FRAGMENT, git(root, "log", "--all", "--format=%B").stdout)
    # -G (regex form) honours -i, so the history search is case-insensitive.
    in_history = len(lines(git(root, "log", "--all", "-i", "-G", NAME_FRAGMENT, "--format=%h").stdout))
    ok = in_files == 0 and in_messages == 0 and in_history == 0
    return line(ok, "names", f"tracked files {in_files}, commit messages {in_messages}, history commits {in_history}")


def check_secrets(root: Path) -> str:
    pattern = "gsk_[A-Za-z0-9]{20,}"
    grep = git(root, "grep", "-c", "-I", "-E", pattern)
    count = sum(int(x.rsplit(":", 1)[1]) for x in lines(grep.stdout))
    history = len(lines(git(root, "log", "--all", "-G", pattern, "--format=%h").stdout))
    return line(count == 0 and history == 0, "secrets",
                f"{count} key-like match(es) in tracked files, {history} history commit(s)")


INVALID_ATTEMPT = "eval_test_attempt1_invalid.json"


def invalid_attempt(results: Path, final: Path) -> tuple[bool, str]:
    """The one allowed extra file (D63): the first final attempt, kept as a disclosure. If present it must
    be valid JSON with split test and be a different file from eval_test.json."""
    path = results / INVALID_ATTEMPT
    if not path.is_file():
        return True, f"{INVALID_ATTEMPT} exists=False"
    try:
        split = json.loads(path.read_text(encoding="utf-8")).get("split")
        valid = True
    except (OSError, json.JSONDecodeError, AttributeError):
        split, valid = None, False
    distinct = not (final.exists() and path.samefile(final))
    ok = valid and split == "test" and distinct
    return ok, f"{INVALID_ATTEMPT} exists=True valid_json={valid} split_test={split == 'test'} distinct={distinct}"


def isolation(results: Path, ids: set[str], after_final: bool) -> tuple[bool, str]:
    """Before the final run eval_test.json must not exist; after it, it must exist, parse and have split test.
    Either way no other file under results/ may contain a test-split id, except the one invalid-attempt
    disclosure file (D63), which must be valid JSON with split test and a different file from eval_test.json."""
    final = results / "eval_test.json"
    exists = final.is_file()
    if after_final:
        try:
            split = json.loads(final.read_text(encoding="utf-8")).get("split") if exists else None
            valid = exists
        except (OSError, json.JSONDecodeError, AttributeError):
            split, valid = None, False
        final_ok = valid and split == "test"
        final_detail = f"eval_test.json exists={exists} valid_json={valid} split_test={split == 'test'}"
    else:
        final_ok = not exists
        final_detail = f"eval_test.json exists={exists}"
    invalid = results / INVALID_ATTEMPT
    invalid_ok, invalid_detail = invalid_attempt(results, final)
    others = [p for p in sorted(results.rglob("*")) if p.is_file() and p not in (final, invalid)] \
        if results.exists() else []
    hits = files_with_ids(others, ids)
    return final_ok and invalid_ok and hits == 0, \
        f"{final_detail}; {invalid_detail}; {hits} of {len(others)} other results file(s) contain a test-split id"


def check_test_isolation(root: Path, after_final: bool = False) -> str:
    gts = json.loads((root / "data" / "v2" / "ground_truth.json").read_text(encoding="utf-8"))
    ids = {g["claim_id"] for g in gts if g.get("split") == "test"}
    ok, detail = isolation(root / "results", ids, after_final)
    return line(ok, "test_isolation", detail)


def check_numbering(root: Path) -> str:
    ok, detail = numbering((root / "DECISIONS.md").read_text(encoding="utf-8"))
    return line(ok, "numbering", detail)


def check_scope(root: Path, base: str) -> str:
    files = changed_files(root, base)
    stat = lines(git(root, "diff", "--stat", base).stdout)
    summary = stat[-1].strip() if stat else "no diff"
    shown = files[:MAX_SCOPE]
    detail = f"{len(files)} file(s) vs {base}; {summary}"
    body = "".join(f"\n  {f}" for f in shown) + ("\n  ..." if len(files) > MAX_SCOPE else "")
    return line(True, "scope", detail + body)


def check_results(root: Path) -> str:
    ok, detail = results_check(root / "results")
    return line(ok, "results", detail)


def main(argv=None, root: Path = ROOT) -> int:
    ap = argparse.ArgumentParser(description="End-of-phase verification.")
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--after-final", action="store_true", help="the one final test run has happened")
    args = ap.parse_args(argv)
    checks = [("head", lambda: check_head(root)), ("tree", lambda: check_tree(root)),
              ("tests", lambda: check_tests(root)), ("protected", lambda: check_protected(root, args.base)),
              ("frozen", lambda: check_frozen(root)), ("names", lambda: check_names(root)),
              ("secrets", lambda: check_secrets(root)), ("test_isolation", lambda: check_test_isolation(root, args.after_final)),
              ("numbering", lambda: check_numbering(root)), ("scope", lambda: check_scope(root, args.base)),
              ("results", lambda: check_results(root))]
    all_ok = True
    for name, check in checks:
        try:
            out = check()
        except Exception as exc:  # a crashing check is a failure, never skipped
            out = line(False, name, f"{type(exc).__name__}: {exc}")
        all_ok &= out.startswith("PASS")
        print(out, flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
