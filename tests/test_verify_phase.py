"""Cheap tests for scripts/verify_phase.py: pure helpers and a tiny tmp git repo. Never runs the full main."""
import importlib.util
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("verify_phase", ROOT / "scripts" / "verify_phase.py")
vp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vp)


def test_numbering_consecutive_duplicate_and_gap():
    good = "- **D1.** a\n- **D2.** b\n\n- **I1** (x): c\n- **I2** (x): d\n"
    assert vp.numbering(good) == (True, "D max 2, I max 2")
    ok, detail = vp.numbering(good + "- **D2.** dup\n")
    assert not ok and "D max 2 (not consecutive from 1)" in detail
    ok, detail = vp.numbering(good.replace("I2", "I3"))
    assert not ok and "I max 3 (not consecutive from 1)" in detail


def test_is_protected():
    assert vp.is_protected("data/v2/claims.json") and vp.is_protected("docs/policies/POL-001.md")
    assert vp.is_protected("src/generate_claims_v2.py") and vp.is_protected("data/v2/TEST_SPLIT.sha256")
    assert not vp.is_protected("docs/README.md") and not vp.is_protected("src/claims_review/rules.py")
    assert not vp.is_protected("database/x.json")


def test_name_fragment_is_not_written_literally_and_counts_case_insensitively():
    assert len(vp.NAME_FRAGMENT) == 7
    assert vp.count_ci(vp.NAME_FRAGMENT, "x " + vp.NAME_FRAGMENT.upper() + " y " + vp.NAME_FRAGMENT) == 2


def test_files_with_ids_counts_files_and_ignores_longer_ids(tmp_path):
    (tmp_path / "a.json").write_text('{"ids": ["V2-0001"]}')
    (tmp_path / "b.json").write_text('{"ids": ["V2-00010"]}')
    (tmp_path / "c.json").write_text("{}")
    paths = sorted(tmp_path.iterdir())
    assert vp.files_with_ids(paths, {"V2-0001"}) == 1
    assert vp.files_with_ids(paths, set()) == 0


def test_results_check_fails_on_dirty_commit(tmp_path):
    assert vp.results_check(tmp_path) == (True, "no eval_*.json files")
    (tmp_path / "eval_dev.json").write_text(json.dumps({"split": "dev", "runs": 3, "claim_count": 20,
                                                        "git_commit": "abc"}))
    ok, detail = vp.results_check(tmp_path)
    assert ok and detail == "eval_dev.json split=dev runs=3 claim_count=20 git_commit=abc"
    (tmp_path / "eval_x.json").write_text(json.dumps({"git_commit": "abc+dirty"}))
    assert vp.results_check(tmp_path)[0] is False


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def test_tree_protected_and_secrets_in_a_tmp_repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.invalid")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "x.json").write_text("{}")
    (tmp_path / "a.py").write_text("x = 1\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "base")
    assert vp.check_tree(tmp_path).startswith("PASS tree")
    assert vp.check_protected(tmp_path, "HEAD").startswith("PASS protected: 0")
    assert vp.check_secrets(tmp_path).startswith("PASS secrets: 0")

    (tmp_path / "data" / "x.json").write_text('{"changed": true}')
    (tmp_path / "new.txt").write_text("key " + "gsk_" + "A" * 24 + "\n")
    assert vp.check_tree(tmp_path) == "FAIL tree: 1 tracked change(s): data/x.json; 1 untracked: new.txt"
    assert vp.check_protected(tmp_path, "HEAD") == "FAIL protected: 1 protected file(s) changed since HEAD: data/x.json"
    _git(tmp_path, "add", "new.txt")
    assert vp.check_secrets(tmp_path) == "FAIL secrets: 1 key-like match(es) in tracked files"
    assert vp.check_protected(tmp_path, "no-such-ref").startswith("FAIL protected: base no-such-ref not found")
