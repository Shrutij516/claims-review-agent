"""Prompt builders for the two LLM calls: the policy judge and the explanation writer."""
from claims_review.codes import CodeTables
from claims_review.models import Flag, ParsedClaim

JUDGE_SYSTEM = (
    "You are a claims review assistant. You judge only whether a clinical note meets the written "
    "policy text you are given. Answer with JSON only, no other text, in exactly this format: "
    '{"flags": [{"flag_id": "...", "criterion_id": "...", "note_quote": "...", "reason": "..."}]}. '
    "Rules: flag_id must be one of the allowed flag ids. criterion_id must be one of the criterion ids "
    "shown in the policy text. note_quote must be copied word for word from the clinical note. reason is "
    "one short sentence. Raise a flag only when the note clearly fails the policy for that flag. "
    'If no allowed flag applies, answer {"flags": []}.'
)

EXPLAIN_SYSTEM = (
    "You write a short explanation of a claim review for a human reviewer. Use only the flags and "
    "evidence you are given. Write at most 120 words of plain text. Cite policy criteria in square "
    "brackets, for example [POL-002.2], and cite only the criterion ids in the allowed list. Do not "
    "mention any other policy or criterion id. A human reviewer makes the decision. Never say a claim is "
    "denied, approved or rejected; say it was flagged for review. Use only facts that appear in the flags, "
    "evidence and criterion text. Do not add codes, policies or numbers that are not given."
)


def _labels(parsed: ParsedClaim, tables: CodeTables) -> tuple[str, list[str]]:
    proc = tables.cpt.get(parsed.cpt) or tables.hcpcs.get(parsed.cpt) or {}
    dx = [f"{i} ({tables.icd10[i]['short_desc']})" if i in tables.icd10 else i for i in parsed.icd10]
    return f"{parsed.cpt} ({proc.get('short_desc', 'unknown code')})", dx


def policy_text(policies) -> str:
    """The full text of the retrieved policies, one criterion per line with its id."""
    if not policies:
        return "(no policy was retrieved)"
    blocks = []
    for p in policies:
        blocks.append(f"{p.policy_id}: {p.title}\n" + "\n".join(f"{cid} {text}" for cid, text in p.criteria))
    return "\n\n".join(blocks)


def judge_messages(parsed: ParsedClaim, tables: CodeTables, policies, allowed_flags: list[dict]) -> list[dict]:
    """Messages for the judge call: claim header, clinical note, policy text, allowed flags."""
    procedure, dx = _labels(parsed, tables)
    flags = "\n".join(f"- {f['flag_id']}: {f['description']}" for f in allowed_flags)
    user = (
        f"CLAIM {parsed.claim_id}\nProcedure: {procedure}\nDiagnoses: {', '.join(dx)}\n"
        f"Modifiers: {', '.join(parsed.modifiers) or 'none'}\n\n"
        f"CLINICAL NOTE\n{parsed.clinical_note}\n\n"
        f"POLICY TEXT\n{policy_text(policies)}\n\n"
        f"ALLOWED FLAGS\n{flags}\n\nAnswer with JSON only."
    )
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def explain_messages(parsed: ParsedClaim, flags: list[Flag], criterion_texts: dict[str, str]) -> list[dict]:
    """Messages for the explanation call: verified flags, their evidence and the cited criterion text."""
    lines = []
    for f in flags:
        lines.append(f"- {f.flag_id}: {f.detail}")
        for e in f.evidence:
            if e["kind"] == "policy":
                lines.append(f"    criterion {e['id']}: {criterion_texts.get(e['id'], '')}")
            elif e["kind"] == "note":
                lines.append(f"    note quote: \"{e['id']}\"")
            elif e["kind"] == "code":
                lines.append(f"    code: {e['id']}")
    allowed = sorted({e["id"] for f in flags for e in f.evidence if e["kind"] == "policy"})
    user = (
        f"CLAIM {parsed.claim_id}\n\nVERIFIED FLAGS\n" + "\n".join(lines)
        + f"\n\nALLOWED CRITERION IDS: {', '.join(allowed) if allowed else '(none)'}"
        + "\n\nWrite the explanation."
    )
    return [{"role": "system", "content": EXPLAIN_SYSTEM}, {"role": "user", "content": user}]
