"""Citation checks for LLM output. Pure functions, no LLM."""
import re

from claims_review.models import ParsedClaim

CRITERION_ID = re.compile(r"POL-\d{3}\.\d+")
BARE_POLICY_ID = re.compile(r"\bPOL-\d{3}\b(?!\.\d)")
DECISION_WORDS = re.compile(r"\b(?:denied|deny|denies|approved|approve|approves|rejected|reject|rejects)\b", re.I)


def collapse(text: str) -> str:
    """Collapse every run of whitespace to one space and trim the ends."""
    return " ".join(text.split())


def _criterion_ids(policies) -> set[str]:
    return {cid for p in policies for cid, _ in p.criteria}


def verify_judge_flags(raw_flags: list, claim: ParsedClaim, policies, catalog: dict) -> tuple[list[dict], list[dict]]:
    """Split judged flags into (verified, rejected).

    A judged flag is {"flag_id", "criterion_id", "note_quote", "reason"}. It is verified only if
    its flag_id is a policy_reading flag in the catalog, its criterion_id is in the text of the
    retrieved policies and belongs to a policy the catalog links to that flag, its note_quote is a non-empty verbatim substring of the clinical note
    (whitespace collapsed), and its flag_id was not already verified. Each rejected entry is
    {"flag": <the raw flag>, "reason": <why>}.
    """
    policies_for_flag = {f["flag_id"]: set(f["policy_ids"]) for f in catalog["flags"] if f["detection"] == "policy_reading"}
    allowed_flags = set(policies_for_flag)
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
        elif criterion_id.split(".")[0] not in policies_for_flag[flag_id]:
            reject(f"criterion_id '{criterion_id}' is not linked to flag '{flag_id}' in the catalog")
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
    """Problems with an explanation: non-empty, only allowed criterion ids, no decision wording.

    Criteria are cited in square brackets, for example [POL-002.2]. A bare policy id such as POL-004
    is a problem unless an allowed id belongs to that policy. Words like denied or approved are a
    problem because the agent only recommends review; a human decides.
    """
    problems = []
    if not isinstance(text, str) or not text.strip():
        return ["explanation is empty"]
    allowed = set(allowed_ids)
    for cid in dict.fromkeys(CRITERION_ID.findall(text)):
        if cid not in allowed:
            problems.append(f"cites {cid}, which is not in the verified evidence")
    for pid in dict.fromkeys(BARE_POLICY_ID.findall(text)):
        if not any(a.startswith(pid + ".") for a in allowed):
            problems.append(f"cites policy {pid} without a criterion, and no verified evidence is in that policy")
    if DECISION_WORDS.search(text):
        problems.append("uses decision wording; the agent only recommends review")
    return problems


def template_explanation(flags) -> str:
    """A deterministic explanation built from flag ids, details and policy evidence. No LLM.

    Inside flag details, any criterion id that is not in the flags' own policy evidence, any bare
    policy id, and any decision word is replaced, so the result passes check_explanation.
    """
    allowed = {e["id"] for f in flags for e in f.evidence if e["kind"] == "policy"}

    def clean(detail: str) -> str:
        detail = CRITERION_ID.sub(lambda m: m.group(0) if m.group(0) in allowed else "[unverified id]", detail)
        detail = BARE_POLICY_ID.sub(lambda m: m.group(0) if any(a.startswith(m.group(0) + ".") for a in allowed)
                                    else "[unverified id]", detail)
        return DECISION_WORDS.sub("[removed]", detail)

    parts = []
    for f in flags:
        cites = " ".join(f"[{e['id']}]" for e in f.evidence if e["kind"] == "policy")
        parts.append(f"{f.flag_id}: {clean(f.detail).strip().rstrip('.')}." + (f" {cites}" if cites else ""))
    return " ".join(parts)
