"""Rules-only baseline: Flagged if any R_ flag fires, otherwise Clear. No LLM."""
from claims_review.models import ClaimResult, ParsedClaim
from claims_review.parsing import parse_claim
from claims_review.rules import rule_flags


def _claim_id(raw, index: int) -> str:
    cid = raw.get("claim_id") if isinstance(raw, dict) else None
    return cid if isinstance(cid, str) and cid.strip() else f"(index {index})"


def run_rules_only(claims: list[dict]) -> list[ClaimResult]:
    """Run the rule checks on raw claim dicts; one result per input claim, failures become Error results."""
    parsed: dict[int, ParsedClaim] = {}
    results: dict[int, ClaimResult] = {}
    for i, raw in enumerate(claims):
        try:
            parsed[i] = parse_claim(raw)
        except Exception as exc:  # any failure is an error result, never dropped (D13)
            results[i] = ClaimResult(_claim_id(raw, i), "Error", [], f"{type(exc).__name__}: {exc}")
    store = list(parsed.values())
    for i, claim in parsed.items():
        try:
            flags = rule_flags(claim, store)
            results[i] = ClaimResult(claim.claim_id, "Flagged" if flags else "Clear", flags)
        except Exception as exc:
            results[i] = ClaimResult(claim.claim_id, "Error", [], f"{type(exc).__name__}: {exc}")
    return [results[i] for i in range(len(claims))]
