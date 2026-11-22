import re
from pathlib import Path

from claims_review.models import Chunk, Hit
from claims_review.retrieval import (Index, build_index, get_criterion, load_chunks, retrieve, tokenize)

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
