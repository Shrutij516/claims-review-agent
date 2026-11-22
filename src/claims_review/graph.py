"""The review agent: a fixed LangGraph state machine.

parse -> rules -> retrieve -> judge -> verify -> explain -> finalize

A claim that cannot be parsed skips straight to finalize with no LLM call. There are at most two
validated LLM calls per claim (the judge, then the explanation), each retried once. No checkpointer
and no human pause yet (Phase 3b).
"""
import json
import operator
import re
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from claims_review.citations import check_explanation, template_explanation, verify_judge_flags
from claims_review.codes import CodeTables, load_tables
from claims_review.models import Flag, ParsedClaim
from claims_review.parsing import ClaimParseError, parse_claim
from claims_review.prompts import explain_messages, judge_messages
from claims_review.retrieval import get_criterion, make_query, retrieve_policies
from claims_review.rules import load_catalog, rule_flags

NODES = ("parse", "rules", "retrieve", "judge", "verify", "explain", "finalize")
NO_ISSUES_TEXT = "No issues found by rule checks or policy reading."
JUDGE_FAILED_TEXT = ("Policy reading failed and no rule check raised a flag, so this claim was not cleared. "
                     "A human must review it.")
JUDGE_FAILED_SUFFIX = " Policy reading failed for this claim, so policy-based issues were not checked."
JUDGE_KEYS = ("flag_id", "criterion_id", "note_quote", "reason")
K_POLICIES = 2
CLIP = 2000


class JudgeFormatError(ValueError):
    """The judge reply was not valid JSON in the required shape."""


class ReviewState(TypedDict, total=False):
    claim_raw: dict
    parsed: ParsedClaim
    status: str                 # ok | llm_failed | parse_error
    judge_status: str           # ok | failed | skipped
    rule_flags: list[Flag]
    policies: list
    judged: list[dict]
    judge_flags: list[Flag]
    rejected: list[dict]
    explanation: str
    explanation_source: str     # llm | template | fixed
    errors: Annotated[list[str], operator.add]
    trace: Annotated[list[dict], operator.add]
    result: dict


def parse_judge_response(text: str) -> list[dict]:
    """Parse and shape-check the judge reply; raise JudgeFormatError if it is unusable."""
    body = (text or "").strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", body, re.S)
    if fenced:
        body = fenced.group(1)
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise JudgeFormatError(f"the reply is not valid JSON ({exc.msg} at position {exc.pos})") from None
    if not isinstance(data, dict) or not isinstance(data.get("flags"), list):
        raise JudgeFormatError('the top level must be an object with a "flags" list')
    out = []
    for i, item in enumerate(data["flags"]):
        if not isinstance(item, dict):
            raise JudgeFormatError(f"flags[{i}] must be an object")
        bad = [k for k in JUDGE_KEYS if not isinstance(item.get(k), str)]
        if bad:
            raise JudgeFormatError(f"flags[{i}] must have string fields: {', '.join(bad)}")
        out.append({k: item[k] for k in JUDGE_KEYS})
    return out


def build_store(raw_claims: list) -> list[ParsedClaim]:
    """Parse a list of raw claims for the cross-claim rules, skipping any that do not parse."""
    store = []
    for raw in raw_claims:
        try:
            store.append(parse_claim(raw))
        except ClaimParseError:
            pass
    return store


def _clip(text) -> str | None:
    return None if text is None else text[:CLIP]


def _entry(node: str, summary: str, data: dict) -> dict:
    return {"node": node, "summary": summary, "data": data}


def _flag_dict(flag: Flag, source: str) -> dict:
    return {"flag_id": flag.flag_id, "source": source, "detail": flag.detail, "evidence": list(flag.evidence)}


def build_graph(llm, store: list[ParsedClaim], tables: CodeTables | None = None):
    """Compile the review graph. store holds the parsed claims of the chosen split (cross-claim rules)."""
    tables = tables or load_tables()
    catalog = load_catalog()
    policy_flags = [f for f in catalog["flags"] if f["detection"] == "policy_reading"]

    def parse(state: ReviewState):
        raw = state["claim_raw"]
        try:
            parsed = parse_claim(raw)
        except ClaimParseError as exc:
            return {"status": "parse_error", "judge_status": "skipped", "errors": [f"parse: {exc}"],
                    "trace": [_entry("parse", "claim could not be parsed; no LLM call", {"error": str(exc)})]}
        return {"parsed": parsed, "status": "ok",
                "trace": [_entry("parse", f"parsed {parsed.claim_id}", {"claim_id": parsed.claim_id})]}

    def route_after_parse(state: ReviewState) -> str:
        return "finalize" if state["status"] == "parse_error" else "rules"

    def rules(state: ReviewState):
        flags = rule_flags(state["parsed"], store, tables)
        return {"rule_flags": flags,
                "trace": [_entry("rules", f"{len(flags)} rule flag(s)",
                                 {"flags": [{"flag_id": f.flag_id, "detail": f.detail} for f in flags]})]}

    def retrieve(state: ReviewState):
        query = make_query(state["parsed"], flags=state["rule_flags"], tables=tables)
        hits = retrieve_policies(query, k=K_POLICIES)
        return {"policies": hits,
                "trace": [_entry("retrieve", f"retrieved {', '.join(h.policy_id for h in hits) or 'no policies'}",
                                 {"query": query, "policies": [{"policy_id": h.policy_id, "score": h.score} for h in hits]})]}

    def judge(state: ReviewState):
        messages = judge_messages(state["parsed"], tables, state["policies"], policy_flags)
        convo, attempts, errors, judged = list(messages), [], [], None
        for attempt in (1, 2):
            reply, problem = None, None
            try:
                reply = llm.complete(convo)
                judged = parse_judge_response(reply)
            except JudgeFormatError as exc:
                problem = str(exc)
            except Exception as exc:  # the LLM call itself failed (network, timeout, rate limit)
                problem = f"LLM call failed: {type(exc).__name__}: {exc}"
            attempts.append({"attempt": attempt, "ok": problem is None, "error": problem, "reply": _clip(reply)})
            if problem is None:
                break
            errors.append(f"judge attempt {attempt}: {problem}")
            if attempt == 1:
                convo = list(messages)
                if reply is not None:
                    convo.append({"role": "assistant", "content": reply})
                convo.append({"role": "user", "content":
                              f"Your answer could not be used: {problem}. Reply again with JSON only, in the exact format required."})
        ok = judged is not None
        update = {"judge_status": "ok" if ok else "failed", "judged": judged or [], "errors": errors,
                  "trace": [_entry("judge", f"judge {'ok' if ok else 'failed'} after {len(attempts)} attempt(s)",
                                   {"attempts": attempts, "judged": judged or []})]}
        if not ok:
            update["status"] = "llm_failed"
        return update

    def verify(state: ReviewState):
        if state["judge_status"] != "ok":
            return {"judge_flags": [], "rejected": [],
                    "trace": [_entry("verify", "skipped because the judge failed", {})]}
        verified, rejected = verify_judge_flags(state["judged"], state["parsed"], state["policies"], catalog)
        flags = [Flag(v["flag_id"], v["reason"], [{"kind": "flag", "id": v["flag_id"]},
                                                   {"kind": "policy", "id": v["criterion_id"]},
                                                   {"kind": "note", "id": v["note_quote"]}]) for v in verified]
        return {"judge_flags": flags, "rejected": rejected,
                "trace": [_entry("verify", f"{len(verified)} verified, {len(rejected)} rejected",
                                 {"verified": verified, "rejected": rejected})]}

    def explain(state: ReviewState):
        flags = state["rule_flags"] + state["judge_flags"]
        failed = state["judge_status"] != "ok"
        if not flags:
            text = JUDGE_FAILED_TEXT if failed else NO_ISSUES_TEXT
            return {"explanation": text, "explanation_source": "fixed",
                    "trace": [_entry("explain", "fixed text, no LLM call", {"text": text})]}

        allowed = {e["id"] for f in flags for e in f.evidence if e["kind"] == "policy"}
        texts = {cid: (get_criterion(cid).text if get_criterion(cid) else "") for cid in allowed}
        messages = explain_messages(state["parsed"], flags, texts)
        convo, attempts, errors, text, source = list(messages), [], [], None, "llm"
        for attempt in (1, 2):
            reply, problems = None, []
            try:
                reply = llm.complete(convo)
                problems = check_explanation(reply, allowed)
            except Exception as exc:
                problems = [f"LLM call failed: {type(exc).__name__}: {exc}"]
            attempts.append({"attempt": attempt, "ok": not problems, "problems": problems, "reply": _clip(reply)})
            if not problems:
                text = reply.strip()
                break
            errors.append(f"explanation attempt {attempt}: {'; '.join(problems)}")
            if attempt == 1:
                convo = list(messages)
                if reply is not None:
                    convo.append({"role": "assistant", "content": reply})
                convo.append({"role": "user", "content":
                              "Problems: " + "; ".join(problems) + ". Rewrite the explanation citing only these ids: "
                              + ", ".join(sorted(allowed)) + "."})
        if text is None:
            text, source = template_explanation(flags), "template"
        if failed:
            text += JUDGE_FAILED_SUFFIX
        return {"explanation": text, "explanation_source": source, "errors": errors,
                "trace": [_entry("explain", f"explanation from {source} after {len(attempts)} attempt(s)",
                                 {"attempts": attempts, "source": source})]}

    def finalize(state: ReviewState):
        raw = state["claim_raw"]
        claim_id = raw.get("claim_id") if isinstance(raw, dict) and isinstance(raw.get("claim_id"), str) else "(no claim_id)"
        status = state["status"]
        if status == "parse_error":
            flags, action = [], "needs_human_review"
            explanation = "The claim could not be parsed, so no review was done. A human must review it."
            source, rejected, policies = "fixed", [], []
        else:
            claim_id = state["parsed"].claim_id
            flags = ([_flag_dict(f, "rule") for f in state["rule_flags"]]
                     + [_flag_dict(f, "policy_reading") for f in state["judge_flags"]])
            action = "flag" if flags else ("pass" if state["judge_status"] == "ok" else "needs_human_review")
            explanation, source = state["explanation"], state["explanation_source"]
            rejected, policies = state["rejected"], [h.policy_id for h in state["policies"]]
        entry = _entry("finalize", f"status {status}, action {action}", {"status": status, "recommended_action": action})
        trace = state["trace"] + [entry]
        result = {"claim_id": claim_id, "status": status, "recommended_action": action, "flags": flags,
                  "explanation": explanation, "explanation_source": source, "rejected": rejected,
                  "retrieved_policies": policies, "errors": list(state["errors"]),
                  "model": getattr(llm, "model", "unknown"), "trace": trace}
        return {"result": result, "trace": [entry]}

    graph = StateGraph(ReviewState)
    for name, fn in zip(NODES, (parse, rules, retrieve, judge, verify, explain, finalize)):
        graph.add_node(name, fn)
    graph.add_edge(START, "parse")
    graph.add_conditional_edges("parse", route_after_parse, {"rules": "rules", "finalize": "finalize"})
    for a, b in (("rules", "retrieve"), ("retrieve", "judge"), ("judge", "verify"), ("verify", "explain"),
                 ("explain", "finalize")):
        graph.add_edge(a, b)
    graph.add_edge("finalize", END)
    return graph.compile()


def run_claim(claim_dict: dict, llm, store: list[ParsedClaim], tables: CodeTables | None = None) -> dict:
    """Review one raw claim dict and return the JSON-serializable result dict."""
    graph = build_graph(llm, store, tables)
    final = graph.invoke({"claim_raw": claim_dict, "errors": [], "trace": []})
    return final["result"]
