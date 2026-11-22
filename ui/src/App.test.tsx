import { fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import type { AgentResult, ClaimSummary, DecisionRecord } from "./api";

const NOTE =
  "Established patient with knee pain for about 4 weeks after increased walking. No physical therapy,  bracing or medication has been tried to date. Exam shows mild joint line tenderness.";
const QUOTE = "No physical therapy, bracing or medication has been tried to date.";

const CLAIMS: ClaimSummary[] = [
  { claim_id: "V2-0051", date_of_service: "2026-02-01", cpt: "99214", procedure: "Office visit, established, level 4", icd10: ["J06.9"], billed_amount: 180, provider: "Dr. A", review_status: "not_started" },
  { claim_id: "V2-0053", date_of_service: "2026-03-12", cpt: "73721", procedure: "MRI lower extremity joint", icd10: ["M25.562"], billed_amount: 633.19, provider: "Dr. Calista Moreau", review_status: "not_started" },
];

const CLAIM = {
  claim_id: "V2-0053", patient_ref: "P-0013", patient_age_band: "60-69", patient_sex: "F", date_of_service: "2026-03-12",
  provider: { name: "Dr. Calista Moreau", specialty: "Orthopedics" }, place_of_service: "22", cpt: "73721",
  modifiers: [], icd10: ["M25.562"], units: 1, billed_amount: 633.19, clinical_note: NOTE,
};

function agentResult(action: AgentResult["recommended_action"] = "flag"): AgentResult {
  return {
    claim_id: "V2-0053", status: "ok", recommended_action: action,
    flags: action === "flag" ? [{
      flag_id: "P_CONSERVATIVE_CARE", source: "policy_reading", detail: "Imaging without a trial of conservative care.",
      evidence: [{ kind: "flag", id: "P_CONSERVATIVE_CARE" }, { kind: "policy", id: "POL-003.1" }, { kind: "note", id: QUOTE }],
    }] : [],
    explanation: "The MRI was ordered before any conservative care [POL-003.1].", explanation_source: "llm",
    rejected: [{ flag: { flag_id: "P_DEGENERATIVE_ONLY", criterion_id: "POL-009.1" }, reason: "criterion POL-009.1 is not in the retrieved text" }],
    retrieved_policies: ["POL-003", "POL-001"], errors: ["judge attempt 1: malformed JSON"], model: "fake-model",
    trace: [
      { node: "parse", summary: "parsed V2-0053", data: { claim_id: "V2-0053" } },
      { node: "rules", summary: "0 rule flag(s)", data: { flags: [] } },
      { node: "retrieve", summary: "retrieved POL-003, POL-001", data: { query: "knee" } },
      { node: "judge", summary: "judge ok after 2 attempt(s)", data: { attempts: [{ attempt: 1, ok: false }, { attempt: 2, ok: true }] } },
      { node: "verify", summary: "1 verified, 1 rejected", data: {} },
      { node: "explain", summary: "explanation from llm after 1 attempt(s)", data: { attempts: [{ attempt: 1, ok: true }] } },
      { node: "finalize", summary: "status ok, action flag", data: { status: "ok" } },
    ],
  };
}

function record(): DecisionRecord {
  const result = agentResult();
  return {
    claim_id: "V2-0053", final_status: "approved",
    human: { action: "approve", final_decision: "flag", reason: null, question: null, reviewer: "reviewer", at: "2026-11-22T10:00:00+00:00" },
    agent_result: { ...result, trace: [...result.trace, { node: "human_review", summary: "reviewer action: approve", data: {} }, { node: "record", summary: "final status approved", data: {} }] },
  };
}

type Handler = (url: string, init?: RequestInit) => { status: number; body: unknown } | Promise<{ status: number; body: unknown }>;
let fetchMock: ReturnType<typeof vi.fn>;
let calls: Array<{ url: string; init?: RequestInit }>;

function mockFetch(handler: Handler) {
  calls = [];
  fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    calls.push({ url, init });
    const { status, body } = await handler(url.replace("http://localhost:8000", ""), init);
    return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
  });
  vi.stubGlobal("fetch", fetchMock);
}

beforeEach(() => {
  window.location.hash = "";
});
afterEach(() => {
  vi.unstubAllGlobals();
});

describe("App", () => {
  it("renders the claim list from /claims with the banner and footer", async () => {
    mockFetch((url) => (url === "/claims" ? { status: 200, body: CLAIMS } : { status: 404, body: { detail: "Claim not found." } }));
    render(<App />);
    const list = await screen.findByRole("navigation", { name: "Claims" });
    expect(await within(list).findByText("V2-0051")).toBeInTheDocument();
    expect(within(list).getByText("V2-0053")).toBeInTheDocument();
    expect(within(list).getByText(/73721 MRI lower extremity joint/)).toBeInTheDocument();
    expect(within(list).getByText("$633.19")).toBeInTheDocument();
    expect(within(list).getAllByText("Not started")).toHaveLength(2);
    expect(screen.getByText("Synthetic data only. Not for real claims.")).toBeInTheDocument();
    expect(screen.getByRole("contentinfo")).toHaveTextContent("claims-review-agent");
  });

  it("runs the full flow: review, evidence, trace, a 422 override, then approve", async () => {
    let state: "not_started" | "awaiting" | "done" = "not_started";
    mockFetch((url, init) => {
      const method = init?.method ?? "GET";
      if (url === "/claims") return { status: 200, body: CLAIMS };
      if (url === "/claims/V2-0053" && method === "GET") return { status: 200, body: { claim: CLAIM, review_status: "not_started" } };
      if (url === "/claims/V2-0053/review" && method === "GET") return { status: 200, body: { status: "not_started" } };
      if (url === "/claims/V2-0053/review" && method === "POST") {
        state = "awaiting";
        const r = agentResult();
        return { status: 201, body: { thread_id: "V2-0053", status: "awaiting_review", agent_result: r, payload: { claim_id: "V2-0053", recommended_action: "flag", flags: r.flags, explanation: r.explanation, errors: r.errors } } };
      }
      if (url === "/criteria/POL-003.1")
        return { status: 200, body: { criterion_id: "POL-003.1", policy_id: "POL-003", policy_title: "Knee MRI", text: "Six weeks of conservative care must be documented." } };
      if (url === "/claims/V2-0053/review/action") {
        const body = JSON.parse(String(init?.body));
        if (body.action === "override" && !body.reason) return { status: 422, body: { problems: ["override needs a non-empty reason"] } };
        state = "done";
        return { status: 200, body: record() };
      }
      return { status: 500, body: { detail: `unexpected ${method} ${url} (${state})` } };
    });

    render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: /V2-0053/ }));
    expect(window.location.hash).toBe("#V2-0053");
    expect(await screen.findByRole("heading", { name: "Claim V2-0053" })).toBeInTheDocument();
    expect(screen.getByText("Dr. Calista Moreau (Orthopedics)")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Run review" }));
    expect(screen.getByRole("status")).toHaveTextContent("The agent is reading the claim (about 10 seconds)");

    expect(await screen.findByText("Flag for review")).toBeInTheDocument();
    expect(screen.getByText(/Explanation written by the model/)).toBeInTheDocument();
    expect(screen.getByText("P_CONSERVATIVE_CARE")).toBeInTheDocument();
    expect(screen.getByText("Policy reading")).toBeInTheDocument();

    // the note quote is highlighted inside the clinical note despite the double space
    const mark = screen.getByTestId("clinical-note").querySelector("mark");
    expect(mark?.textContent).toBe("No physical therapy,  bracing or medication has been tried to date.");

    // the policy chip loads the criterion text on click
    fireEvent.click(screen.getByRole("button", { name: "POL-003.1" }));
    expect(await screen.findByText("Six weeks of conservative care must be documented.")).toBeInTheDocument();
    expect(calls.some((c) => c.url.endsWith("/criteria/POL-003.1"))).toBe(true);

    // rejected citations and the trace timeline
    expect(screen.getByText("Rejected by checks (1)")).toBeInTheDocument();
    const trace = screen.getByRole("list", { name: "Reasoning trace" });
    const nodes = Array.from(trace.querySelectorAll("[data-node]")).map((n) => n.getAttribute("data-node"));
    expect(nodes).toEqual(["parse", "rules", "retrieve", "judge", "verify", "explain", "finalize"]);
    expect(within(trace).getByText("Retried (2 attempts)")).toBeInTheDocument();

    // override with no reason: the client only hints, the server answers 422 with problems
    expect(screen.getByLabelText("Reviewer name")).toHaveValue("reviewer");
    fireEvent.click(screen.getByLabelText("Override"));
    fireEvent.change(screen.getByLabelText("Decision"), { target: { value: "pass" } });
    expect(screen.getByText("A reason is required.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Submit" }));
    expect(await screen.findByText("override needs a non-empty reason")).toBeInTheDocument();
    expect(screen.getByText("The server did not accept this action:")).toBeInTheDocument();

    // approve: the Decision record card replaces the panel
    fireEvent.click(screen.getByLabelText("Approve the recommendation"));
    fireEvent.click(screen.getByRole("button", { name: "Submit" }));
    const card = await screen.findByRole("region", { name: "Decision record" });
    expect(within(card).getByText("Approved")).toBeInTheDocument();
    expect(within(card).getByText("Approve")).toBeInTheDocument();
    expect(within(card).getByText("Flag")).toBeInTheDocument();
    expect(within(card).getByText("Flag for review")).toBeInTheDocument();
    expect(within(card).getByText("reviewer")).toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "Human review" })).not.toBeInTheDocument();
    const traceNodes = Array.from(screen.getByRole("list", { name: "Reasoning trace" }).querySelectorAll("[data-node]")).map((n) => n.getAttribute("data-node"));
    expect(traceNodes.slice(-2)).toEqual(["human_review", "record"]);
    expect(window.localStorage.getItem("claims-review.reviewer")).toBe("reviewer");
  });

  it("disables Approve when the agent recommends needs_human_review", async () => {
    window.location.hash = "#V2-0053";
    mockFetch((url) => {
      if (url === "/claims") return { status: 200, body: CLAIMS };
      if (url === "/claims/V2-0053") return { status: 200, body: { claim: CLAIM, review_status: "awaiting_review" } };
      const r = agentResult("needs_human_review");
      return { status: 200, body: { status: "awaiting_review", agent_result: r, payload: { claim_id: "V2-0053", recommended_action: r.recommended_action, flags: [], explanation: r.explanation, errors: [] } } };
    });
    render(<App />);
    expect(await screen.findByText("Needs human review")).toBeInTheDocument();
    expect(screen.getByLabelText("Approve the recommendation")).toBeDisabled();
    expect(screen.getByText(/Approve is not available/)).toBeInTheDocument();
    expect(screen.getByLabelText("Override")).toBeChecked();
  });

  it("shows the server message when the API key is missing (503)", async () => {
    window.location.hash = "#V2-0053";
    mockFetch((url, init) => {
      if (url === "/claims") return { status: 200, body: CLAIMS };
      if (url === "/claims/V2-0053") return { status: 200, body: { claim: CLAIM, review_status: "not_started" } };
      if (init?.method === "POST") return { status: 503, body: { detail: "The environment variable GROQ_API_KEY is not set." } };
      return { status: 200, body: { status: "not_started" } };
    });
    render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: "Run review" }));
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("The server cannot run a review: The environment variable GROQ_API_KEY is not set.");
  });

  it("says the API may not be running when fetch fails", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("Failed to fetch"); }));
    render(<App />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Is the API running?");
  });

  it("explains an incomplete review needs the stored review deleted", async () => {
    window.location.hash = "#V2-0053";
    mockFetch((url) => {
      if (url === "/claims") return { status: 200, body: CLAIMS };
      if (url === "/claims/V2-0053") return { status: 200, body: { claim: CLAIM, review_status: "incomplete" } };
      return { status: 200, body: { status: "incomplete" } };
    });
    render(<App />);
    expect(await screen.findByRole("alert")).toHaveTextContent("the stored review must be deleted");
    expect(screen.queryByRole("button", { name: "Run review" })).not.toBeInTheDocument();
  });
});
