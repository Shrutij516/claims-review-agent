"""FastAPI layer over the review agent and human review. It serves the DEV split only.

Run (from the repo root):
    .venv/bin/uvicorn --factory --app-dir src claims_review.api:create_app

The test split is never loaded into memory, so it cannot be reached through the API. Ground-truth
fields are never returned: the ground truth file is read only to choose which claims are dev claims.
The LLM is built only when a review is started, so a missing GROQ_API_KEY never breaks startup.
"""
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from claims_review import DATA_V2, ROOT
from claims_review.codes import load_tables
from claims_review.graph import build_store
from claims_review.llm import GroqLLM, MissingAPIKeyError
from claims_review.retrieval import get_criterion, get_policy
from claims_review.review import (ReviewActionError, ReviewExistsError, close_saver, get_review, open_saver,
                                  start_review, submit_review)

SPLIT = "dev"
DEFAULT_DB = "runs/checkpoints.sqlite"
ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
NOT_FOUND = "Claim not found."
CRITERION_NOT_FOUND = "Criterion not found."


class ActionBody(BaseModel):
    """The reviewer action. Every field is optional here so that content problems (a missing reviewer,
    a missing reason) come back as the same {"problems": [...]} list as every other action problem."""
    model_config = ConfigDict(extra="forbid")
    action: str | None = None
    reviewer: str | None = None
    final_decision: str | None = None
    reason: str | None = None
    question: str | None = None


def _load_dev_claims() -> dict[str, dict]:
    """Read claims.json and keep only the dev claims; ground truth is used only to pick the split."""
    claims = json.loads((DATA_V2 / "claims.json").read_text(encoding="utf-8"))
    split_of = {g["claim_id"]: g["split"]
                for g in json.loads((DATA_V2 / "ground_truth.json").read_text(encoding="utf-8"))}
    return {c["claim_id"]: c for c in claims if split_of[c["claim_id"]] == SPLIT}


def create_app(llm_factory: Callable[[], object] | None = None, db_path: str | Path = DEFAULT_DB,
               clock: Callable[[], object] | None = None) -> FastAPI:
    """Build the app. llm_factory is called only when a review starts; tests pass one that returns a fake."""
    llm_factory = llm_factory or GroqLLM
    db = Path(db_path)
    db = db if db.is_absolute() else ROOT / db
    claims = _load_dev_claims()
    store = build_store(list(claims.values()))
    tables = load_tables()

    app = FastAPI(title="Claims review agent (dev split)", version="0.1.0")
    app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["*"], allow_headers=["*"],
                       allow_credentials=False)

    @contextmanager
    def saver_scope():
        saver = open_saver(db)
        try:
            yield saver
        finally:
            close_saver(saver)

    def claim_or_404(claim_id: str) -> dict:
        claim = claims.get(claim_id)
        if claim is None:
            raise HTTPException(status_code=404, detail=NOT_FOUND)
        return claim

    def status_of(saver, claim_id: str) -> str:
        status = get_review(claim_id, saver)["status"]
        return "not_started" if status == "not_found" else status

    def procedure_label(cpt: str) -> str | None:
        row = tables.cpt.get(cpt) or tables.hcpcs.get(cpt)
        return row["short_desc"] if row else None

    @app.get("/health")
    def health():
        return {"status": "ok", "split": SPLIT}

    @app.get("/criteria/{criterion_id}")
    def read_criterion(criterion_id: str):
        """Policy text for one criterion id (for example POL-001.1). Policy text only, never ground truth."""
        chunk = get_criterion(criterion_id)
        if chunk is None:
            raise HTTPException(status_code=404, detail=CRITERION_NOT_FOUND)
        return {"criterion_id": chunk.criterion_id, "policy_id": chunk.policy_id,
                "policy_title": get_policy(chunk.policy_id).title, "text": chunk.text}

    @app.get("/claims")
    def list_claims():
        with saver_scope() as saver:
            return [{"claim_id": cid, "date_of_service": c["date_of_service"], "cpt": c["cpt"],
                     "procedure": procedure_label(c["cpt"]), "icd10": c["icd10"],
                     "billed_amount": c["billed_amount"], "provider": c["provider"]["name"],
                     "review_status": status_of(saver, cid)}
                    for cid, c in sorted(claims.items())]

    @app.get("/claims/{claim_id}")
    def read_claim(claim_id: str):
        claim = claim_or_404(claim_id)
        with saver_scope() as saver:
            return {"claim": claim, "review_status": status_of(saver, claim_id)}

    @app.post("/claims/{claim_id}/review", status_code=201)
    def start(claim_id: str):
        claim = claim_or_404(claim_id)
        with saver_scope() as saver:
            if status_of(saver, claim_id) != "not_started":
                raise HTTPException(status_code=409, detail=f"A review for claim {claim_id} already exists "
                                                            "(one review per claim).")
            try:
                llm = llm_factory()
            except MissingAPIKeyError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from None
            try:
                return JSONResponse(status_code=201, content=start_review(claim, llm, store, saver, tables))
            except ReviewExistsError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from None

    @app.get("/claims/{claim_id}/review")
    def read_review(claim_id: str):
        claim_or_404(claim_id)
        with saver_scope() as saver:
            review = get_review(claim_id, saver)
        if review["status"] == "not_found":
            return {"status": "not_started"}
        return review

    @app.post("/claims/{claim_id}/review/action")
    def act(claim_id: str, body: ActionBody = Body(...)):
        claim_or_404(claim_id)
        with saver_scope() as saver:
            status = status_of(saver, claim_id)
            if status != "awaiting_review":
                raise HTTPException(status_code=409, detail=f"No review is waiting for this claim (status: {status}).")
            try:
                return submit_review(claim_id, body.model_dump(exclude_none=True), saver, now=clock)
            except ReviewActionError as exc:
                return JSONResponse(status_code=422, content={"problems": exc.problems})

    return app
