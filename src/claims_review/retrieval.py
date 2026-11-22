"""BM25 retrieval over one chunk per numbered policy criterion."""
import re
from functools import lru_cache
from pathlib import Path

from rank_bm25 import BM25Okapi

from claims_review import POLICY_DIR
from claims_review.models import Chunk, Hit

STOPWORDS = frozenset(
    "a an and are as at be been by for from has have in is it its of on or that the this to was were will with".split()
)
_TOKEN = re.compile(r"[a-z0-9]+")
_CRITERION = re.compile(r"^(POL-\d{3})\.(\d+)\s+(.*)$", re.M)
_TITLE = re.compile(r"^#\s+POL-\d{3}:\s*(.*)$", re.M)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens with a short stopword list removed."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


def load_chunks(policy_dir: Path | None = None) -> list[Chunk]:
    """Split docs/policies/POL-*.md into one Chunk per numbered criterion."""
    chunks = []
    for path in sorted((policy_dir or POLICY_DIR).glob("POL-*.md")):
        text = path.read_text(encoding="utf-8")
        title = _TITLE.search(text)
        for m in _CRITERION.finditer(text):
            chunks.append(Chunk(policy_id=m.group(1), criterion_id=f"{m.group(1)}.{m.group(2)}",
                                text=m.group(3).strip(), title=title.group(1).strip() if title else ""))
    return sorted(chunks, key=lambda c: c.criterion_id)


class Index:
    """A BM25 index over chunks; the title is indexed with the criterion text."""

    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("cannot build an index with no chunks")
        self.chunks = list(chunks)
        self._by_id = {c.criterion_id: c for c in self.chunks}
        self._bm25 = BM25Okapi([tokenize(f"{c.title} {c.text}") for c in self.chunks])

    def get(self, criterion_id: str) -> Chunk | None:
        """Return the chunk for a criterion id, or None."""
        return self._by_id.get(criterion_id)

    def search(self, query: str, k: int) -> list[Hit]:
        """Top k positive-score hits; ties are broken by criterion id."""
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.chunks, scores), key=lambda p: (-float(p[1]), p[0].criterion_id))
        return [Hit(c.policy_id, c.criterion_id, c.text, float(s)) for c, s in ranked[:k] if s > 0]


@lru_cache(maxsize=None)
def build_index(policy_dir: Path | None = None) -> Index:
    """Build (and cache) the index over the policy files."""
    return Index(load_chunks(policy_dir))


def retrieve(query: str, k: int = 3, index: Index | None = None) -> list[Hit]:
    """Return up to k policy criteria for a query, best first."""
    return (index or build_index()).search(query, k)


def get_criterion(criterion_id: str, index: Index | None = None) -> Chunk | None:
    """Look up a criterion by id, for citation checks."""
    return (index or build_index()).get(criterion_id)
