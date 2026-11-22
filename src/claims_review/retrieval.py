"""BM25 retrieval over policy criteria (one chunk each) and over whole policies (one document each)."""
import re
from functools import lru_cache
from pathlib import Path

from rank_bm25 import BM25Okapi

from claims_review import POLICY_DIR
from claims_review.codes import CodeTables, load_tables
from claims_review.models import Chunk, Flag, Hit, ParsedClaim, Policy, PolicyHit
from claims_review.rules import load_catalog

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


# ------------------------------------------------------------------ whole-policy retrieval
def _criterion_number(criterion_id: str) -> int:
    return int(criterion_id.rsplit(".", 1)[1])


def load_policies(policy_dir: Path | None = None) -> list[Policy]:
    """One Policy per policy file, with every criterion in numeric order, sorted by policy id."""
    grouped: dict[str, list[Chunk]] = {}
    for c in load_chunks(policy_dir):
        grouped.setdefault(c.policy_id, []).append(c)
    return [Policy(pid, chunks[0].title,
                   tuple((c.criterion_id, c.text) for c in sorted(chunks, key=lambda c: _criterion_number(c.criterion_id))))
            for pid, chunks in sorted(grouped.items())]


class PolicyIndex:
    """A BM25 index with one document per whole policy (title plus all criteria text)."""

    def __init__(self, policies: list[Policy]):
        if not policies:
            raise ValueError("cannot build an index with no policies")
        self.policies = list(policies)
        self._by_id = {p.policy_id: p for p in self.policies}
        docs = [tokenize(p.title + " " + " ".join(text for _, text in p.criteria)) for p in self.policies]
        self._bm25 = BM25Okapi(docs)

    def get(self, policy_id: str) -> Policy:
        """Return a policy by id, or raise ValueError naming the known ids."""
        try:
            return self._by_id[policy_id]
        except KeyError:
            raise ValueError(f"unknown policy id '{policy_id}'; known ids: {', '.join(sorted(self._by_id))}") from None

    def search(self, query: str, k: int) -> list[PolicyHit]:
        """Top k positive-score policies; ties are broken by policy id."""
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.policies, scores), key=lambda p: (-float(p[1]), p[0].policy_id))
        return [PolicyHit(p.policy_id, p.title, float(s), p.criteria) for p, s in ranked[:k] if s > 0]


@lru_cache(maxsize=None)
def build_policy_index(policy_dir: Path | None = None) -> PolicyIndex:
    """Build (and cache) the whole-policy index over the policy files."""
    return PolicyIndex(load_policies(policy_dir))


def retrieve_policies(query: str, k: int = 2, index: PolicyIndex | None = None) -> list[PolicyHit]:
    """Return up to k whole policies for a query, best first, each with all its criteria in order."""
    return (index or build_policy_index()).search(query, k)


def get_policy(policy_id: str, index: PolicyIndex | None = None) -> Policy:
    """Look up a whole policy by id; raises ValueError for an unknown id."""
    return (index or build_policy_index()).get(policy_id)


def make_query(claim: ParsedClaim, flags: list[Flag] | None = None, tables: CodeTables | None = None) -> str:
    """Retrieval query: procedure label, diagnosis labels and clinical note.

    When flags are given, each flag id and its catalog description are appended.
    """
    tables = tables or load_tables()
    proc = tables.cpt.get(claim.cpt) or tables.hcpcs.get(claim.cpt) or {}
    parts = [proc.get("short_desc", "")]
    parts += [tables.icd10[i]["short_desc"] for i in claim.icd10 if i in tables.icd10]
    parts.append(claim.clinical_note)
    if flags:
        descriptions = {f["flag_id"]: f["description"] for f in load_catalog()["flags"]}
        for flag in flags:
            if flag.flag_id not in descriptions:
                raise ValueError(f"flag id '{flag.flag_id}' is not in the flag catalog")
            parts += [flag.flag_id, descriptions[flag.flag_id]]
    return " ".join(p for p in parts if p)
