import json

import pytest
from fastapi.testclient import TestClient

from claims_review.api import create_app
from claims_review.llm import MissingAPIKeyError
from conftest import FakeLLM

NO_FLAGS = json.dumps({"flags": []})
EXPLAIN = "A level 4 visit with only a tier 1 diagnosis was flagged for review [POL-002.2] [POL-002.4]."
DEV_ID = "V2-0051"        # a dev claim with one rule flag (R_EM_LEVEL_VS_DX)
TEST_ID = "V2-0001"       # a claim id from the frozen test split; the API must not know it
FIXED_TIME = "2026-11-22T10:00:00+00:00"
GROUND_TRUTH_KEYS = {"expected_flags", "is_anomalous", "category", "needs_policy_reading", "notes", "split"}
APPROVE = {"action": "approve", "reviewer": "alex"}


class Factory:
    """An llm_factory that returns a fresh scripted fake each time and counts how often it is called."""

    def __init__(self, *replies):
        self.replies, self.count, self.llms = replies or (NO_FLAGS, EXPLAIN), 0, []

    def __call__(self):
        self.count += 1
        self.llms.append(FakeLLM(*self.replies))
        return self.llms[-1]


@pytest.fixture
def db(tmp_path):
    return tmp_path / "runs" / "checkpoints.sqlite"


def make_client(db, factory=None, clock=None):
    return TestClient(create_app(llm_factory=factory or Factory(), db_path=db, clock=clock))


def keys_at_any_depth(value):
    if isinstance(value, dict):
        for k, v in value.items():
            yield k
            yield from keys_at_any_depth(v)
    elif isinstance(value, list):
        for item in value:
            yield from keys_at_any_depth(item)


# ------------------------------------------------------------------ read endpoints
def test_health(db):
    r = make_client(db).get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok", "split": "dev"}


def test_claims_lists_only_dev_claims_with_the_listed_fields(db):
    r = make_client(db).get("/claims")
    assert r.status_code == 200
    items = r.json()
    assert len(items) == 20
    ids = [i["claim_id"] for i in items]
    assert ids == sorted(ids) and TEST_ID not in ids and DEV_ID in ids
    for item in items:
        assert set(item) == {"claim_id", "date_of_service", "cpt", "procedure", "icd10", "billed_amount",
                             "provider", "review_status"}
        assert isinstance(item["provider"], str) and isinstance(item["icd10"], list)
        assert item["review_status"] == "not_started"
    item = next(i for i in items if i["claim_id"] == DEV_ID)
    assert item["cpt"] == "99214" and item["procedure"]


def test_claim_detail_returns_the_raw_claim_with_its_note(db):
    r = make_client(db).get(f"/claims/{DEV_ID}")
    assert r.status_code == 200
    body = r.json()
    assert body["review_status"] == "not_started"
    assert body["claim"]["claim_id"] == DEV_ID and body["claim"]["clinical_note"]
    assert set(body["claim"]) >= {"cpt", "icd10", "modifiers", "units", "billed_amount", "provider"}


def test_a_test_split_claim_is_indistinguishable_from_an_unknown_one_on_every_route(db):
    client = make_client(db)
    routes = [("get", "/claims/{}", None), ("post", "/claims/{}/review", None), ("get", "/claims/{}/review", None),
              ("post", "/claims/{}/review/action", APPROVE)]
    for method, path, body in routes:
        test_r = getattr(client, method)(path.format(TEST_ID), **({"json": body} if body else {}))
        unknown_r = getattr(client, method)(path.format("NOPE-9999"), **({"json": body} if body else {}))
        assert test_r.status_code == unknown_r.status_code == 404, path
        assert test_r.json() == unknown_r.json() == {"detail": "Claim not found."}, path


def test_startup_and_reads_never_need_an_api_key_or_call_the_llm_factory(db, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    client = TestClient(create_app(db_path=db))            # default factory builds GroqLLM lazily
    assert client.get("/health").status_code == 200 and client.get("/claims").status_code == 200
    factory = Factory()
    client = make_client(db, factory)
    client.get("/claims"), client.get(f"/claims/{DEV_ID}"), client.get(f"/claims/{DEV_ID}/review")
    assert factory.count == 0


def test_criterion_endpoint_returns_policy_text_for_a_known_id(db):
    r = make_client(db).get("/criteria/POL-001.1")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"criterion_id", "policy_id", "policy_title", "text"}
    assert body["criterion_id"] == "POL-001.1" and body["policy_id"] == "POL-001"
    assert body["policy_title"] and body["text"].startswith("Diagnosis specificity.")


@pytest.mark.parametrize("criterion_id", ["POL-001.99", "POL-999.1", "nope"])
def test_criterion_endpoint_is_404_for_an_unknown_id(db, criterion_id):
    r = make_client(db).get(f"/criteria/{criterion_id}")
    assert r.status_code == 404 and r.json() == {"detail": "Criterion not found."}


# ------------------------------------------------------------------ the full flow
def test_full_review_flow_and_no_ground_truth_in_any_response(db):
    factory = Factory()
    client = make_client(db, factory, clock=lambda: FIXED_TIME)
    bodies = []

    def status_in_list():
        return next(i for i in client.get("/claims").json() if i["claim_id"] == DEV_ID)["review_status"]

    r = client.get(f"/claims/{DEV_ID}/review")
    assert r.status_code == 200 and r.json() == {"status": "not_started"} and status_in_list() == "not_started"
    bodies.append(r.json())

    r = client.post(f"/claims/{DEV_ID}/review")
    assert r.status_code == 201
    started = r.json()
    bodies.append(started)
    assert set(started) == {"thread_id", "status", "payload", "agent_result"}
    assert started["thread_id"] == DEV_ID and started["status"] == "awaiting_review"
    assert started["payload"]["recommended_action"] == "flag"
    assert [t["node"] for t in started["agent_result"]["trace"]][:2] == ["parse", "rules"]
    assert started["agent_result"]["flags"][0]["flag_id"] == "R_EM_LEVEL_VS_DX"
    assert factory.count == 1 and len(factory.llms[0].calls) == 2 and status_in_list() == "awaiting_review"

    r = client.get(f"/claims/{DEV_ID}/review")
    waiting = r.json()
    bodies.append(waiting)
    assert r.status_code == 200 and waiting["status"] == "awaiting_review"
    assert waiting["payload"] == started["payload"] and waiting["agent_result"] == started["agent_result"]
    bodies.append(client.get(f"/claims/{DEV_ID}").json())
    assert bodies[-1]["review_status"] == "awaiting_review"

    r = client.post(f"/claims/{DEV_ID}/review/action", json=APPROVE)
    assert r.status_code == 200
    record = r.json()
    bodies.append(record)
    assert record["final_status"] == "approved" and record["claim_id"] == DEV_ID
    assert record["human"]["at"] == FIXED_TIME and record["human"]["reviewer"] == "alex"
    assert len(factory.llms[0].calls) == 2          # the action made no LLM call

    r = client.get(f"/claims/{DEV_ID}/review")
    assert r.status_code == 200 and r.json() == {"status": "completed", "record": record}
    bodies.append(r.json())
    assert status_in_list() == "completed"
    bodies.append(client.get("/claims").json())
    bodies.append(client.get("/criteria/POL-002.4").json())
    bodies.append(client.get("/criteria/POL-999.1").json())

    for body in bodies:
        assert not (set(keys_at_any_depth(body)) & GROUND_TRUTH_KEYS), body


def test_an_error_response_never_contains_ground_truth_keys_either(db):
    client = make_client(db)
    client.post(f"/claims/{DEV_ID}/review")
    for r in (client.post(f"/claims/{DEV_ID}/review"), client.get(f"/claims/{TEST_ID}"),
              client.post(f"/claims/{DEV_ID}/review/action", json={"action": "override", "reviewer": "a"})):
        assert not (set(keys_at_any_depth(r.json())) & GROUND_TRUTH_KEYS)


# ------------------------------------------------------------------ error paths
def test_override_without_a_reason_is_422_with_problems_and_the_review_stays_awaiting(db):
    client = make_client(db)
    client.post(f"/claims/{DEV_ID}/review")
    r = client.post(f"/claims/{DEV_ID}/review/action",
                    json={"action": "override", "reviewer": "alex", "final_decision": "pass"})
    assert r.status_code == 422
    assert list(r.json()) == ["problems"] and any("reason" in p for p in r.json()["problems"])
    assert client.get(f"/claims/{DEV_ID}/review").json()["status"] == "awaiting_review"
    assert client.post(f"/claims/{DEV_ID}/review/action", json=APPROVE).status_code == 200


def test_a_missing_reviewer_is_a_problems_list_too(db):
    client = make_client(db)
    client.post(f"/claims/{DEV_ID}/review")
    r = client.post(f"/claims/{DEV_ID}/review/action", json={"action": "approve"})
    assert r.status_code == 422 and any("reviewer" in p for p in r.json()["problems"])


def test_action_before_start_and_after_completion_is_409_and_a_second_start_is_409(db):
    client = make_client(db)
    r = client.post(f"/claims/{DEV_ID}/review/action", json=APPROVE)
    assert r.status_code == 409 and "not_started" in r.json()["detail"]
    assert client.post(f"/claims/{DEV_ID}/review").status_code == 201
    r = client.post(f"/claims/{DEV_ID}/review")
    assert r.status_code == 409 and "already exists" in r.json()["detail"]
    assert client.post(f"/claims/{DEV_ID}/review/action", json=APPROVE).status_code == 200
    r = client.post(f"/claims/{DEV_ID}/review/action", json=APPROVE)
    assert r.status_code == 409 and "completed" in r.json()["detail"]
    assert client.post(f"/claims/{DEV_ID}/review").status_code == 409


def test_a_second_start_does_not_call_the_llm_factory_again(db):
    factory = Factory()
    client = make_client(db, factory)
    client.post(f"/claims/{DEV_ID}/review")
    client.post(f"/claims/{DEV_ID}/review")
    assert factory.count == 1


def test_missing_key_is_503_names_the_variable_and_leaves_no_review_behind(db, monkeypatch):
    def no_key():
        raise MissingAPIKeyError("The environment variable GROQ_API_KEY is not set.")

    client = make_client(db, no_key)
    r = client.post(f"/claims/{DEV_ID}/review")
    assert r.status_code == 503 and "GROQ_API_KEY" in r.json()["detail"]
    assert client.get(f"/claims/{DEV_ID}/review").json() == {"status": "not_started"}

    secret = "gsk-test-NOT-A-REAL-KEY-12345"
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    real = TestClient(create_app(db_path=db))             # the default factory with no key set
    r = real.post(f"/claims/{DEV_ID}/review")
    assert r.status_code == 503 and "GROQ_API_KEY" in r.json()["detail"] and secret not in r.text


def test_an_unknown_field_in_the_action_body_is_422(db):
    client = make_client(db)
    client.post(f"/claims/{DEV_ID}/review")
    r = client.post(f"/claims/{DEV_ID}/review/action", json={**APPROVE, "is_anomalous": True})
    assert r.status_code == 422
    assert client.get(f"/claims/{DEV_ID}/review").json()["status"] == "awaiting_review"


def test_request_info_over_the_api(db):
    client = make_client(db)
    client.post(f"/claims/{DEV_ID}/review")
    r = client.post(f"/claims/{DEV_ID}/review/action",
                    json={"action": "request_info", "reviewer": "alex", "question": "Which diagnosis?"})
    assert r.status_code == 200 and r.json()["final_status"] == "needs_info"


# ------------------------------------------------------------------ CORS and restart
@pytest.mark.parametrize("origin", ["http://localhost:5173", "http://127.0.0.1:5173"])
def test_cors_preflight_from_the_vite_dev_server_is_allowed(db, origin):
    r = make_client(db).options(f"/claims/{DEV_ID}/review/action", headers={
        "Origin": origin, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"})
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == origin
    assert "POST" in r.headers["access-control-allow-methods"]
    assert "access-control-allow-credentials" not in r.headers


def test_cors_preflight_from_another_origin_is_not_allowed(db):
    r = make_client(db).options("/claims", headers={"Origin": "http://evil.example",
                                                    "Access-Control-Request-Method": "GET"})
    assert r.status_code == 400 and "access-control-allow-origin" not in r.headers
    r = make_client(db).get("/claims", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in r.headers


def test_a_second_app_on_the_same_db_file_sees_the_review_and_the_record(db):
    first = make_client(db)
    started = first.post(f"/claims/{DEV_ID}/review").json()
    second = make_client(db)
    again = second.get(f"/claims/{DEV_ID}/review").json()
    assert again["status"] == "awaiting_review" and again["agent_result"] == started["agent_result"]
    assert next(i for i in second.get("/claims").json() if i["claim_id"] == DEV_ID)["review_status"] == "awaiting_review"
    record = second.post(f"/claims/{DEV_ID}/review/action", json=APPROVE).json()
    third = make_client(db)
    assert third.get(f"/claims/{DEV_ID}/review").json() == {"status": "completed", "record": record}
    assert third.post(f"/claims/{DEV_ID}/review").status_code == 409
