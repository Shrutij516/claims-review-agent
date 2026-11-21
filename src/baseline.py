"""
Baseline claims classifier: no LLM, no reasoning. Parses the raw claim line
with a fixed regex, checks it against the same policy table used by the
agent (rules.py), and applies fixed if/else logic: any rule violation at
all -> "Flagged", otherwise "Clean". This is the comparison point for the
agentic pipeline in agent.py.
"""
import re
import time

from rules import rule_check

LINE_PATTERN = re.compile(
    r"^(?P<claim_id>\S+) \| DOS: (?P<dos>\S+) \| Provider: (?P<provider>.+?) \((?P<specialty>.+?)\) \| "
    r"CPT: (?P<cpt>\S+) \| ICD-10: (?P<icd>\S+) \| Billed: \$(?P<billed>[\d.]+) \| Units: (?P<units>\d+)$"
)


def parse_raw_note(raw_note):
    m = LINE_PATTERN.match(raw_note.strip())
    if not m:
        raise ValueError(f"Could not parse claim line: {raw_note!r}")
    d = m.groupdict()
    return {
        "claim_id": d["claim_id"],
        "date_of_service": d["dos"],
        "provider_name": d["provider"],
        "specialty": d["specialty"],
        "cpt_code": d["cpt"],
        "icd10_code": d["icd"],
        "billed_amount": float(d["billed"]),
        "units": int(d["units"]),
    }


def classify(raw_note):
    """Returns (result_dict, latency_seconds)."""
    start = time.perf_counter()
    claim = parse_raw_note(raw_note)
    flags = rule_check(claim)
    violated_rules = [name for name in ("amount_outlier", "upcoding", "mismatch") if flags[name]]
    decision = "Flagged" if flags["any_flag"] else "Clean"
    latency = time.perf_counter() - start
    result = {
        "claim_id": claim["claim_id"],
        "decision": decision,
        "violated_rules": violated_rules,
        "flags": flags,
    }
    return result, latency
