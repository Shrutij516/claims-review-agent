"""Citation checks for LLM output. Pure functions, no LLM."""
import re

from claims_review.models import ParsedClaim

CRITERION_ID = re.compile(r"POL-\d{3}\.\d+")


def collapse(text: str) -> str:
    """Collapse every run of whitespace to one space and trim the ends."""
    return " ".join(text.split())


def _criterion_ids(policies) -> set[str]:
    return {cid for p in policies for cid, _ in p.criteria}


def verify_judge_flags(raw_flags: list, claim: ParsedClaim, policies, catalog: dict) -> tuple[list[dict], list[dict]]:
    """Split judged flags into (verified, rejected).

    A judged flag is {"flag_id", "criterion_id", "note_quote", "reason"}. It is verified only if
    its flag_id is a policy_reading flag in the catalog, its criterion_id is in the text of the
    retrieved policies, its note_quote is a non-empty verbatim substring of the clinical note
    (whitespace collapsed), and its flag_id was not already verified. Each rejected entry is
    {"flag": <the raw flag>, "reason": <why>}.
    """
    allowed_flags = {f["flag_id"] for f in catalog["flags"] if f["detection"] == "policy_reading"}
    allowed_criteria = _criterion_ids(policies)
    note = collapse(claim.clinical_note)
    verified, rejected, seen = [], [], set()

    for raw in raw_flags:
        def reject(reason, raw=raw):
            rejected.append({"flag": raw, "reason": reason})

        if not isinstance(raw, dict):
            reject("judged flag is not an object")
            continue
        flag_id, criterion_id = raw.get("flag_id"), raw.get("criterion_id")
        quote, reason_text = raw.get("note_quote"), raw.get("reason")
        if not all(isinstance(v, str) for v in (flag_id, criterion_id, quote, reason_text)):
            reject("flag_id, criterion_id, note_quote and reason must all be strings")
        elif flag_id not in allowed_flags:
            reject(f"flag_id '{flag_id}' is not a policy_reading flag in the catalog")
        elif criterion_id not in allowed_criteria:
            reject(f"criterion_id '{criterion_id}' is not in the retrieved policy text")
        elif not collapse(quote):
            reject("note_quote is empty")
        elif collapse(quote) not in note:
            reject("note_quote is not a verbatim substring of the clinical note")
        elif flag_id in seen:
            reject(f"duplicate flag_id '{flag_id}'")
        else:
            seen.add(flag_id)
            verified.append({"flag_id": flag_id, "criterion_id": criterion_id,
                             "note_quote": collapse(quote), "reason": reason_text.strip()})
    return verified, rejected


def check_explanation(text: str, allowed_ids) -> list[str]:
    """Problems with an explanation: it must be non-empty and cite only allowed criterion ids.

    Criteria are cited in square brackets, for example [POL-002.2].
    """
    problems = []
    if not isinstance(text, str) or not text.strip():
        return ["explanation is empty"]
    allowed = set(allowed_ids)
    for cid in dict.fromkeys(CRITERION_ID.findall(text)):
        if cid not in allowed:
            problems.append(f"cites {cid}, which is not in the verified evidence")
    return problems


def template_explanation(flags) -> str:
    """A deterministic explanation built from flag ids, details and policy evidence. No LLM.

    Any criterion id inside a flag detail that is not in that flag set's own policy evidence is
    replaced, so the result always passes check_explanation.
    """
    allowed = {e["id"] for f in flags for e in f.evidence if e["kind"] == "policy"}

    def clean(detail: str) -> str:
        return CRITERION_ID.sub(lambda m: m.group(0) if m.group(0) in allowed else "[unverified id]", detail)

    parts = []
    for f in flags:
        cites = " ".join(f"[{e['id']}]" for e in f.evidence if e["kind"] == "policy")
        parts.append(f"{f.flag_id}: {clean(f.detail).strip().rstrip('.')}." + (f" {cites}" if cites else ""))
    return " ".join(parts)
