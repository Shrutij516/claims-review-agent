"""Human review on top of the agent graph: pause with interrupt(), persist in SQLite, resume with an action.

One review per claim: the LangGraph thread_id is the claim_id. Resuming re-runs no node before
human_review, so it makes no LLM call and survives a process restart.
"""
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Callable

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from claims_review.codes import CodeTables
from claims_review.graph import build_graph
from claims_review.models import ParsedClaim

ACTIONS = ("approve", "override", "request_info")
DECISIONS = ("pass", "flag")


class ReviewExistsError(RuntimeError):
    """A review for this claim already exists (one review per claim)."""


class ReviewActionError(ValueError):
    """The reviewer action is not valid, or there is no review waiting for it."""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


def open_saver(path) -> SqliteSaver:
    """Open (creating folders and file if needed) a SQLite checkpoint saver usable across threads."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))


def close_saver(saver: SqliteSaver) -> None:
    """Close the saver's SQLite connection."""
    saver.conn.close()


def _text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_review_action(action, recommended_action: str) -> list[str]:
    """Problems with a reviewer action; an empty list means it is valid.

    approve is allowed only when the agent recommended pass or flag. override needs a final
    decision (pass or flag) and a reason. request_info needs a question. Every action needs a reviewer.
    """
    if not isinstance(action, dict):
        return ["action must be an object"]
    problems = []
    kind = action.get("action")
    if kind not in ACTIONS:
        problems.append(f"action must be one of {', '.join(ACTIONS)}")
    if not _text(action.get("reviewer")):
        problems.append("reviewer must be a non-empty string")
    if kind == "approve" and recommended_action not in DECISIONS:
        problems.append(f"approve is not allowed when the agent recommended '{recommended_action}'; "
                        "override with a decision or request more information")
    if kind == "override":
        if action.get("final_decision") not in DECISIONS:
            problems.append(f"override needs final_decision to be one of {', '.join(DECISIONS)}")
        if not _text(action.get("reason")):
            problems.append("override needs a non-empty reason")
    if kind == "request_info" and not _text(action.get("question")):
        problems.append("request_info needs a non-empty question")
    return problems


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _inspect_graph(saver: SqliteSaver, clock=None):
    """A graph used only to read stored state (no LLM, no store): no node runs."""
    return build_graph(None, [], checkpointer=saver, human_review=True, clock=clock)


def _pending_payload(snapshot) -> dict | None:
    """The interrupt payload if the thread is paused at human_review, else None."""
    if tuple(snapshot.next) != ("human_review",):
        return None
    for task in snapshot.tasks:
        for interrupt_ in task.interrupts:
            return interrupt_.value
    return None


def start_review(claim_dict: dict, llm, store: list[ParsedClaim], saver: SqliteSaver,
                 tables: CodeTables | None = None) -> dict:
    """Run the agent on a claim and pause for a human. The thread_id is the claim_id."""
    thread_id = claim_dict.get("claim_id") if isinstance(claim_dict, dict) else None
    if not isinstance(thread_id, str) or not thread_id.strip():
        raise ValueError("the claim has no claim_id, so a review cannot be started for it")
    config = _config(thread_id)
    if saver.get_tuple(config) is not None:
        raise ReviewExistsError(f"A review for claim {thread_id} already exists (one review per claim). "
                                "Use show to see it or submit to finish it.")
    graph = build_graph(llm, store, tables, checkpointer=saver, human_review=True)
    graph.invoke({"claim_raw": claim_dict, "errors": [], "trace": []}, config)
    payload = _pending_payload(graph.get_state(config))
    if payload is None:
        raise RuntimeError(f"review of claim {thread_id} did not pause for a human")
    return {"thread_id": thread_id, "status": "awaiting_review", "payload": payload}


def get_review(thread_id: str, saver: SqliteSaver) -> dict:
    """Read a review without running anything: not_found, awaiting_review (payload) or completed (record)."""
    config = _config(thread_id)
    if saver.get_tuple(config) is None:
        return {"status": "not_found"}
    snapshot = _inspect_graph(saver).get_state(config)
    payload = _pending_payload(snapshot)
    if payload is not None:
        return {"status": "awaiting_review", "payload": payload}
    if "record" in snapshot.values:
        return {"status": "completed", "record": snapshot.values["record"]}
    return {"status": "incomplete"}


def submit_review(thread_id: str, action: dict, saver: SqliteSaver,
                  now: Callable[[], datetime | str] | None = None) -> dict:
    """Validate a reviewer action against the stored payload, then resume and return the final record.

    A bad action raises ReviewActionError listing every problem, and the review stays awaiting_review.
    """
    review = get_review(thread_id, saver)
    if review["status"] != "awaiting_review":
        raise ReviewActionError([f"no review is waiting for claim {thread_id} (status: {review['status']})"])
    problems = validate_review_action(action, review["payload"]["recommended_action"])
    if problems:
        raise ReviewActionError(problems)
    graph = _inspect_graph(saver, clock=now)
    final = graph.invoke(Command(resume=action), _config(thread_id))
    return final["record"]
