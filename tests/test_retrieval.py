import re
from pathlib import Path

import pytest

from claims_review.codes import load_tables
from claims_review.models import Chunk, Flag, Hit, Policy, PolicyHit
from claims_review.retrieval import (Index, PolicyIndex, build_index, get_criterion, get_policy, load_chunks,
                                     load_policies, make_query, retrieve, retrieve_policies, tokenize)
from claims_review.rules import load_catalog
from factories import make

ROOT = Path(__file__).resolve().parent.parent


def test_one_chunk_per_numbered_criterion():
    expected = []
    for path in sorted((ROOT / "docs" / "policies").glob("POL-*.md")):
        expected += re.findall(r"^(POL-\d{3}\.\d+)\s", path.read_text(encoding="utf-8"), flags=re.M)
    chunks = load_chunks()
    assert sorted(c.criterion_id for c in chunks) == sorted(expected)
    assert len(chunks) == len(set(expected)) > 20
    for c in chunks:
        assert c.criterion_id.startswith(c.policy_id + ".") and c.text and c.title


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The Knee MRI, of 6 weeks!") == ["knee", "mri", "6", "weeks"]


def test_retrieve_returns_hits_with_existing_criterion_ids():
    hits = retrieve("knee MRI requires six weeks of conservative therapy physical therapy", k=3)
    assert 1 <= len(hits) <= 3
    assert all(isinstance(h, Hit) for h in hits)
    for h in hits:
        chunk = get_criterion(h.criterion_id)
        assert chunk is not None and chunk.text == h.text and chunk.policy_id == h.policy_id
        assert h.score > 0
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_obvious_query_finds_the_right_policy_first():
    top = retrieve("knee MRI requires six weeks of conservative therapy physical therapy", k=3)[0]
    assert top.policy_id == "POL-003"


def test_k_is_respected_and_unknown_words_give_no_hits():
    assert len(retrieve("knee MRI therapy", k=1)) == 1
    assert retrieve("zzzzqqqq xxyyzz") == []
    assert retrieve("") == []


def test_retrieval_is_deterministic_across_calls_and_fresh_indexes():
    q = "right knee meniscus tear arthroscopic partial meniscectomy degenerative findings"
    first = retrieve(q, k=5)
    assert retrieve(q, k=5) == first
    fresh = Index(load_chunks())
    assert retrieve(q, k=5, index=fresh) == first


def test_ties_are_broken_by_criterion_id():
    chunks = [
        Chunk("POL-900", "POL-900.2", "alpha beta", "T"),
        Chunk("POL-900", "POL-900.1", "alpha beta", "T"),
        Chunk("POL-901", "POL-901.1", "gamma delta", "T"),
        Chunk("POL-902", "POL-902.1", "epsilon zeta", "T"),
        Chunk("POL-903", "POL-903.1", "eta theta", "T"),
    ]
    hits = retrieve("alpha", k=5, index=Index(chunks))
    assert [h.criterion_id for h in hits] == ["POL-900.1", "POL-900.2"]
    assert hits[0].score == hits[1].score


def test_get_criterion_unknown_id_is_none():
    assert get_criterion("POL-999.9") is None
    assert get_criterion("POL-003.2") is not None


def test_build_index_is_cached():
    assert build_index() is build_index()


def _stray_lines(text: str) -> list[str]:
    """Non-empty lines in the Criteria section that are not a POL-xxx.n criterion line."""
    lines = text.splitlines()
    start = lines.index("## Criteria")
    return [ln for ln in lines[start + 1:] if ln.strip() and not re.match(r"^POL-\d{3}\.\d+\s", ln)]


def test_stray_line_check_catches_a_wrapped_criterion():
    wrapped = "## Criteria\n\nPOL-009.1 First part of a criterion\nthat continues here.\n\nPOL-009.2 Fine.\n"
    assert _stray_lines(wrapped) == ["that continues here."]
    assert _stray_lines("## Criteria\n\nPOL-009.1 Fine.\n\nPOL-009.2 Fine.\n") == []


def test_every_policy_criterion_is_on_a_single_line():
    """load_chunks reads only the first line of a criterion, so nothing else may sit in the Criteria section."""
    for path in sorted((ROOT / "docs" / "policies").glob("POL-*.md")):
        assert _stray_lines(path.read_text(encoding="utf-8")) == [], path.name


# ------------------------------------------------------------------ whole-policy retrieval
def _criteria_in_file_order(path):
    return [(m.group(1), m.group(2).strip())
            for m in re.finditer(r"^(POL-\d{3}\.\d+)\s+(.*)$", path.read_text(encoding="utf-8"), flags=re.M)]


def test_load_policies_returns_every_criterion_of_every_policy_in_file_order():
    policies = load_policies()
    files = sorted((ROOT / "docs" / "policies").glob("POL-*.md"))
    assert [p.policy_id for p in policies] == [f.stem for f in files] and len(policies) == 8
    for policy, path in zip(policies, files):
        assert list(policy.criteria) == _criteria_in_file_order(path)
        assert policy.title


def test_retrieve_policies_returns_whole_policies_with_all_criteria_in_order():
    hits = retrieve_policies("knee MRI requires six weeks of conservative therapy physical therapy")
    assert 1 <= len(hits) <= 2 and all(isinstance(h, PolicyHit) for h in hits)
    assert hits[0].policy_id == "POL-003"
    for h in hits:
        assert list(h.criteria) == list(get_policy(h.policy_id).criteria) and h.score > 0
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_retrieve_policies_is_deterministic_across_calls_and_fresh_indexes():
    q = "right knee meniscus tear arthroscopic partial meniscectomy degenerative findings"
    first = retrieve_policies(q, k=3)
    assert retrieve_policies(q, k=3) == first
    assert retrieve_policies(q, k=3, index=PolicyIndex(load_policies())) == first


def test_retrieve_policies_respects_k_and_returns_nothing_for_unknown_words():
    assert len(retrieve_policies("knee MRI therapy", k=1)) == 1
    assert retrieve_policies("zzzzqqqq xxyyzz") == []


def test_policy_ties_are_broken_by_policy_id():
    pols = [Policy("POL-902", "T", (("POL-902.1", "alpha beta"),)), Policy("POL-901", "T", (("POL-901.1", "alpha beta"),)),
            Policy("POL-903", "T", (("POL-903.1", "gamma delta"),)), Policy("POL-904", "T", (("POL-904.1", "eta theta"),)),
            Policy("POL-905", "T", (("POL-905.1", "iota kappa"),))]
    hits = retrieve_policies("alpha", k=5, index=PolicyIndex(pols))
    assert [h.policy_id for h in hits] == ["POL-901", "POL-902"] and hits[0].score == hits[1].score


def test_get_policy_returns_the_policy_and_unknown_id_raises_a_clear_error():
    assert get_policy("POL-003").policy_id == "POL-003" and len(get_policy("POL-003").criteria) == 4
    with pytest.raises(ValueError, match=r"unknown policy id 'POL-999'.*POL-001.*POL-008"):
        get_policy("POL-999")


def test_make_query_plain_has_procedure_diagnosis_and_note_but_no_flag_text():
    claim = make(cpt="73721", icd10=["M25.561"], clinical_note="Knee pain after a fall.")
    t = load_tables()
    q = make_query(claim)
    assert t.cpt["73721"]["short_desc"] in q and t.icd10["M25.561"]["short_desc"] in q
    assert "Knee pain after a fall." in q and "R_" not in q
    assert make_query(claim, flags=[]) == q


def test_make_query_with_flags_appends_each_flag_id_and_catalog_description():
    claim = make(cpt="73721", icd10=["M25.569"])
    flags = [Flag("R_LATERALITY_MISSING", "x"), Flag("R_UNITS_EXCEEDED", "y")]
    q = make_query(claim, flags=flags)
    descriptions = {f["flag_id"]: f["description"] for f in load_catalog()["flags"]}
    for f in flags:
        assert f.flag_id in q and descriptions[f.flag_id] in q
    assert q.startswith(make_query(claim))


def test_make_query_unknown_flag_id_raises():
    with pytest.raises(ValueError, match="R_NOPE"):
        make_query(make(), flags=[Flag("R_NOPE", "x")])
