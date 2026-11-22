// Typed client for the claims review API. fetch only; the server is authoritative for every decision.

export const API_URL: string = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/+$/, "");

export type ReviewStatus = "not_started" | "awaiting_review" | "completed" | "incomplete";
export type RecommendedAction = "flag" | "pass" | "needs_human_review";
export type FinalDecision = "pass" | "flag";
export type ActionKind = "approve" | "override" | "request_info";
export type FinalStatus = "approved" | "overridden" | "needs_info";

export interface ClaimSummary {
  claim_id: string;
  date_of_service: string;
  cpt: string;
  procedure: string | null;
  icd10: string[];
  billed_amount: number;
  provider: string;
  review_status: ReviewStatus;
}

export interface Claim {
  claim_id: string;
  patient_ref: string;
  patient_age_band: string;
  patient_sex: string;
  date_of_service: string;
  provider: { name: string; specialty: string };
  place_of_service: string;
  cpt: string;
  modifiers: string[];
  icd10: string[];
  units: number;
  billed_amount: number;
  clinical_note: string;
}

export interface ClaimDetail {
  claim: Claim;
  review_status: ReviewStatus;
}

export interface Evidence {
  kind: "flag" | "policy" | "note" | string;
  id: string;
}

export interface Flag {
  flag_id: string;
  source: "rule" | "policy_reading";
  detail: string;
  evidence: Evidence[];
}

export interface Rejected {
  flag: Record<string, unknown>;
  reason: string;
}

export interface TraceEntry {
  node: string;
  summary: string;
  data: Record<string, unknown>;
}

export interface AgentResult {
  claim_id: string;
  status: string;
  recommended_action: RecommendedAction;
  flags: Flag[];
  explanation: string;
  explanation_source: "llm" | "template" | "fixed";
  rejected: Rejected[];
  retrieved_policies: string[];
  errors: string[];
  model: string;
  trace: TraceEntry[];
}

export interface Payload {
  claim_id: string;
  recommended_action: RecommendedAction;
  flags: Flag[];
  explanation: string;
  errors: string[];
}

export interface HumanAction {
  action: ActionKind;
  final_decision: FinalDecision | null;
  reason: string | null;
  question: string | null;
  reviewer: string;
  at: string;
}

export interface DecisionRecord {
  claim_id: string;
  agent_result: AgentResult;
  human: HumanAction;
  final_status: FinalStatus;
}

export type Review =
  | { status: "not_started" }
  | { status: "incomplete" }
  | { status: "awaiting_review"; payload: Payload; agent_result: AgentResult; thread_id?: string }
  | { status: "completed"; record: DecisionRecord };

export interface Criterion {
  criterion_id: string;
  policy_id: string;
  policy_title: string;
  text: string;
}

export interface ActionBody {
  action: ActionKind;
  reviewer: string;
  final_decision?: FinalDecision;
  reason?: string;
  question?: string;
}

/** Any failure talking to the API. status 0 means the request never reached the server. */
export class ApiError extends Error {
  status: number;
  detail: string | null;
  problems: string[] | null;

  constructor(status: number, message: string, detail: string | null = null, problems: string[] | null = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
    this.problems = problems;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, init);
  } catch {
    throw new ApiError(0, `Cannot reach the API at ${API_URL}. Is the API running?`);
  }
  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    body = null;
  }
  if (!res.ok) {
    const obj = (body && typeof body === "object" ? body : {}) as { detail?: unknown; problems?: unknown };
    const detail = typeof obj.detail === "string" ? obj.detail : null;
    const problems = Array.isArray(obj.problems) ? obj.problems.map(String) : null;
    throw new ApiError(res.status, detail ?? `Request failed with status ${res.status}.`, detail, problems);
  }
  return body as T;
}

const jsonPost = (body?: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: body === undefined ? undefined : JSON.stringify(body),
});

export const api = {
  listClaims: () => request<ClaimSummary[]>("/claims"),
  getClaim: (id: string) => request<ClaimDetail>(`/claims/${encodeURIComponent(id)}`),
  getReview: (id: string) => request<Review>(`/claims/${encodeURIComponent(id)}/review`),
  startReview: (id: string) =>
    request<{ thread_id: string; status: "awaiting_review"; payload: Payload; agent_result: AgentResult }>(
      `/claims/${encodeURIComponent(id)}/review`,
      jsonPost(),
    ),
  submitAction: (id: string, body: ActionBody) =>
    request<DecisionRecord>(`/claims/${encodeURIComponent(id)}/review/action`, jsonPost(body)),
  getCriterion: (id: string) => request<Criterion>(`/criteria/${encodeURIComponent(id)}`),
};
