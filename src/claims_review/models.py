"""Dataclasses shared across the claims_review package."""
from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True)
class ParsedClaim:
    """A validated claim with normalized codes and a real date."""
    claim_id: str
    patient_ref: str
    patient_age_band: str
    patient_sex: str
    date_of_service: date
    provider_name: str
    provider_specialty: str
    place_of_service: str
    cpt: str
    modifiers: tuple[str, ...]
    icd10: tuple[str, ...]
    units: int
    billed_amount: float
    clinical_note: str


@dataclass(frozen=True)
class CodeCheck:
    """Result of looking one code up in the code tables."""
    system: str
    code: str
    valid: bool
    description: str | None
    laterality: str | None


@dataclass(frozen=True)
class Flag:
    """A raised flag with a human readable detail and citable evidence ids."""
    flag_id: str
    detail: str
    evidence: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class Chunk:
    """One numbered policy criterion, the unit of retrieval."""
    policy_id: str
    criterion_id: str
    text: str
    title: str = ""


@dataclass(frozen=True)
class Hit:
    """A retrieved criterion with its BM25 score."""
    policy_id: str
    criterion_id: str
    text: str
    score: float


@dataclass(frozen=True)
class ClaimResult:
    """Decision for one claim: Flagged, Clear or Error (never dropped)."""
    claim_id: str
    decision: str
    flags: list[Flag] = field(default_factory=list)
    error: str | None = None
