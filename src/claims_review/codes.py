"""Code tables (ICD-10, CPT, HCPCS, modifiers) and per-claim code validation."""
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from claims_review import DATA_CODES
from claims_review.models import CodeCheck, ParsedClaim


@dataclass(frozen=True)
class CodeTables:
    """The four lookup tables, each mapping code to its table row."""
    icd10: dict[str, dict]
    cpt: dict[str, dict]
    hcpcs: dict[str, dict]
    modifiers: dict[str, dict]


@lru_cache(maxsize=None)
def load_tables(codes_dir: Path | None = None) -> CodeTables:
    """Load the code tables from data/codes (or another folder)."""
    base = codes_dir or DATA_CODES

    def rows(name: str, key: str = "codes") -> dict[str, dict]:
        data = json.loads((base / name).read_text(encoding="utf-8"))
        return {row["code"]: row for row in data[key]}

    return CodeTables(
        icd10=rows("icd10.json"),
        cpt=rows("cpt.json"),
        hcpcs=rows("hcpcs.json"),
        modifiers=rows("cpt.json", key="modifiers"),
    )


def validate_codes(parsed: ParsedClaim, tables: CodeTables | None = None) -> list[CodeCheck]:
    """Check the procedure code, modifiers and diagnosis codes; the procedure comes first."""
    tables = tables or load_tables()
    checks: list[CodeCheck] = []

    if parsed.cpt in tables.cpt:
        checks.append(CodeCheck("cpt", parsed.cpt, True, tables.cpt[parsed.cpt]["short_desc"], None))
    elif parsed.cpt in tables.hcpcs:
        checks.append(CodeCheck("hcpcs", parsed.cpt, True, tables.hcpcs[parsed.cpt]["short_desc"], None))
    else:
        checks.append(CodeCheck("cpt_or_hcpcs", parsed.cpt, False, None, None))

    for m in parsed.modifiers:
        row = tables.modifiers.get(m)
        checks.append(CodeCheck("modifier", m, row is not None, row["short_desc"] if row else None, None))

    for i in parsed.icd10:
        row = tables.icd10.get(i)
        checks.append(CodeCheck("icd10", i, row is not None,
                                row["short_desc"] if row else None,
                                row["laterality"] if row else None))
    return checks
