"""The review agent: a fixed LangGraph state machine.

parse -> rules -> retrieve -> judge -> verify -> explain -> finalize
                                                          (-> human_review -> record, when human_review=True)

A claim that cannot be parsed skips straight to finalize with no LLM call. There are at most two
validated LLM calls per claim (the judge, then the explanation), each retried once.

Everything stored in the graph state is a plain JSON-friendly value (dicts, lists, strings, numbers),
so a SQLite checkpoint round trip needs no custom serialization. Objects such as Flag and
ParsedClaim are rebuilt inside the nodes that need them.
"""
import json
import operator
import re
from datetime import datetime, timezone
from typing import Annotated, Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from claims_review.citations import check_explanation, template_explanation, verify_judge_flags
from claims_review.codes import CodeTables, load_tables
from claims_review.models import Flag, ParsedClaim, PolicyHit
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
FINAL_STATUS = {"approve": "approved", "override": "overridden", "request_info": "needs_info"}
K_POLICIES = 2
CLIP = 2000


class JudgeFormatError(ValueError):
    """The judge reply was not valid JSON in the required shape."""


class ReviewState(TypedDict, total=False):
    claim_raw: dict
    status: str                 # ok | llm_failed | parse_error
    judge_status: str           # ok | failed | skipped
    rule_flags: list[dict]      # {"flag_id", "detail", "evidence"}
    policies: list[dict]        # {"policy_id", "title", "score", "criteria": [[id, text], ...]}
    judged: list[dict]
    judge_flags: list[dict]
    rejected: list[dict]
    explanation: str
    explanation_source: str     # llm | template | fixed
    errors: Annotated[list[str], operator.add]
    trace: Annotated[list[dict], operator.add]
    result: dict
    human_action: dict
    record: dict


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


def _flag_to_dict(flag: Flag) -> dict:
    return {"flag_id": flag.flag_id, "detail": flag.detail, "evidence": [dict(e) for e in flag.evidence]}


def _flag_from_dict(d: dict) -> Flag:
    return Flag(d["flag_id"], d["detail"], [dict(e) for e in d["evidence"]])


def _policy_hits(items: list[dict]) -> list[PolicyHit]:
    return [PolicyHit(p["policy_id"], p["title"], p["score"], tuple((c[0], c[1]) for c in p["criteria"]))
            for p in items]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def build_graph(llm, store: list[ParsedClaim], tables: CodeTables | None = None,
                checkpointer=None, human_review: bool = False, clock: Callable[[], datetime | str] | None = None):
    """Compile the review graph.

    store holds the parsed claims of the chosen split (for the cross-claim rules). With human_review
    False the graph ends at finalize. With human_review True it continues finalize -> human_review ->
    record and needs a checkpointer, because human_review pauses with interrupt(). clock returns the
    time stamped on the human record (a datetime or an ISO string); tests pass a fixed one.
    """
    if human_review and checkpointer is None:
        raise ValueError("human_review=True needs a checkpointer, because the graph pauses with interrupt()")
    tables = tables or load_tables()
    catalog = load_catalog()
    policy_flags = [f for f in catalog["flags"] if f["detection"] == "policy_reading"]
    clock = clock or _utc_now

    def parse(state: ReviewState):
        raw = state["claim_raw"]
        try:
            parsed = parse_claim(raw)
        except ClaimParseError as exc:
            return {"status": "parse_error", "judge_status": "skipped", "errors": [f"parse: {exc}"],
                    "trace": [_entry("parse", "claim could not be parsed; no LLM call", {"error": str(exc)})]}
        return {"status": "ok",
                "trace": [_entry("parse", f"parsed {parsed.claim_id}", {"claim_id": parsed.claim_id})]}

    def route_after_parse(state: ReviewState) -> str:
        return "finalize" if state["status"] == "parse_error" else "rules"

    def rules(state: ReviewState):
        flags = rule_flags(parse_claim(state["claim_raw"]), store, tables)
        return {"rule_flags": [_flag_to_dict(f) for f in flags],
                "trace": [_entry("rules", f"{len(flags)} rule flag(s)",
                                 {"flags": [{"flag_id": f.flag_id, "detail": f.detail} for f in flags]})]}

    def retrieve(state: ReviewState):
        flags = [_flag_from_dict(d) for d in state["rule_flags"]]
        query = make_query(parse_claim(state["claim_raw"]), flags=flags, tables=tables)
        hits = retrieve_policies(query, k=K_POLICIES)
        return {"policies": [{"policy_id": h.policy_id, "title": h.title, "score": h.score,
                              "criteria": [[cid, text] for cid, text in h.criteria]} for h in hits],
                "trace": [_entry("retrieve", f"retrieved {', '.join(h.policy_id for h in hits) or 'no policies'}",
                                 {"query": query, "policies": [{"policy_id": h.policy_id, "score": h.score} for h in hits]})]}

    def judge(state: ReviewState):
        messages = judge_messages(parse_claim(state["claim_raw"]), tables, _policy_hits(state["policies"]), policy_flags)
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
        verified, rejected = verify_judge_flags(state["judged"], parse_claim(state["claim_raw"]),
                                                _policy_hits(state["policies"]), catalog)
        flags = [{"flag_id": v["flag_id"], "detail": v["reason"],
                  "evidence": [{"kind": "flag", "id": v["flag_id"]}, {"kind": "policy", "id": v["criterion_id"]},
                               {"kind": "note", "id": v["note_quote"]}]} for v in verified]
        return {"judge_flags": flags, "rejected": rejected,
                "trace": [_entry("verify", f"{len(verified)} verified, {len(rejected)} rejected",
                                 {"verified": verified, "rejected": rejected})]}

    def explain(state: ReviewState):
        flags = [_flag_from_dict(d) for d in state["rule_flags"] + state["judge_flags"]]
        failed = state["judge_status"] != "ok"
        if not flags:
            text = JUDGE_FAILED_TEXT if failed else NO_ISSUES_TEXT
            return {"explanation": text, "explanation_source": "fixed",
                    "trace": [_entry("explain", "fixed text, no LLM call", {"text": text})]}

        allowed = {e["id"] for f in flags for e in f.evidence if e["kind"] == "policy"}
        texts = {cid: (get_criterion(cid).text if get_criterion(cid) else "") for cid in allowed}
        messages = explain_messages(parse_claim(state["claim_raw"]), flags, texts)
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
            claim_id = parse_claim(raw).claim_id
            flags = [{"flag_id": d["flag_id"], "source": source_name, "detail": d["detail"], "evidence": d["evidence"]}
                     for source_name, items in (("rule", state["rule_flags"]), ("policy_reading", state["judge_flags"]))
                     for d in items]
            action = "flag" if flags else ("pass" if state["judge_status"] == "ok" else "needs_human_review")
            explanation, source = state["explanation"], state["explanation_source"]
            rejected, policies = state["rejected"], [p["policy_id"] for p in state["policies"]]
        entry = _entry("finalize", f"status {status}, action {action}", {"status": status, "recommended_action": action})
        trace = state["trace"] + [entry]
        result = {"claim_id": claim_id, "status": status, "recommended_action": action, "flags": flags,
                  "explanation": explanation, "explanation_source": source, "rejected": rejected,
                  "retrieved_policies": policies, "errors": list(state["errors"]),
                  "model": getattr(llm, "model", "unknown"), "trace": trace}
        return {"result": result, "trace": [entry]}

    def human_review_node(state: ReviewState):
        # No side effects before interrupt(): on resume LangGraph runs this node again from the top.
        result = state["result"]
        payload = {"claim_id": result["claim_id"], "recommended_action": result["recommended_action"],
                   "flags": result["flags"], "explanation": result["explanation"], "errors": result["errors"]}
        action = interrupt(payload)
        name = action.get("action") if isinstance(action, dict) else None
        return {"human_action": action,
                "trace": [_entry("human_review", f"reviewer action: {name}", {"action": action})]}

    def record_node(state: ReviewState):
        action, result = state["human_action"], state["result"]
        stamp = clock()
        kind = action["action"]
        final_decision = result["recommended_action"] if kind == "approve" else action.get("final_decision")
        human = {"action": kind, "final_decision": final_decision if kind != "request_info" else None,
                 "reason": action.get("reason"), "question": action.get("question"),
                 "reviewer": action["reviewer"].strip(), "at": stamp if isinstance(stamp, str) else stamp.isoformat()}
        final_status = FINAL_STATUS[kind]
        entry = _entry("record", f"final status {final_status}", {"final_status": final_status})
        rec = {"claim_id": result["claim_id"], "agent_result": {**result, "trace": state["trace"] + [entry]},
               "human": human, "final_status": final_status}
        return {"record": rec, "trace": [entry]}

    graph = StateGraph(ReviewState)
    nodes = dict(zip(NODES, (parse, rules, retrieve, judge, verify, explain, finalize)))
    for name, fn in nodes.items():
        graph.add_node(name, fn)
    graph.add_edge(START, "parse")
    graph.add_conditional_edges("parse", route_after_parse, {"rules": "rules", "finalize": "finalize"})
    for a, b in (("rules", "retrieve"), ("retrieve", "judge"), ("judge", "verify"), ("verify", "explain"),
                 ("explain", "finalize")):
        graph.add_edge(a, b)
    if human_review:
        graph.add_node("human_review", human_review_node)
        graph.add_node("record", record_node)
        graph.add_edge("finalize", "human_review")
        graph.add_edge("human_review", "record")
        graph.add_edge("record", END)
    else:
        graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)


def run_claim(claim_dict: dict, llm, store: list[ParsedClaim], tables: CodeTables | None = None) -> dict:
    """Review one raw claim dict and return the JSON-serializable result dict."""
    graph = build_graph(llm, store, tables)
    final = graph.invoke({"claim_raw": claim_dict, "errors": [], "trace": []})
    return final["result"]
