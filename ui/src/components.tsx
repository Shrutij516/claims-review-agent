import { useState, type FormEvent } from "react";
import {
  ApiError,
  type ActionBody,
  type ActionKind,
  type AgentResult,
  type Claim,
  type ClaimSummary,
  type Criterion,
  type DecisionRecord,
  type Evidence,
  type FinalDecision,
  type Flag,
  type RecommendedAction,
  type ReviewStatus,
  type TraceEntry,
} from "./api";
import { highlightQuotes } from "./highlight";

// ------------------------------------------------------------------ labels and formatting
export const STATUS_LABEL: Record<ReviewStatus, string> = {
  not_started: "Not started",
  awaiting_review: "Awaiting review",
  completed: "Completed",
  incomplete: "Incomplete",
};

export const RECOMMENDATION_LABEL: Record<RecommendedAction, string> = {
  flag: "Flag for review",
  pass: "No issues found",
  needs_human_review: "Needs human review",
};

const FINAL_STATUS_LABEL: Record<string, string> = {
  approved: "Approved",
  overridden: "Overridden",
  needs_info: "More information requested",
};

const ACTION_LABEL: Record<ActionKind, string> = {
  approve: "Approve",
  override: "Override",
  request_info: "Request more information",
};

const DECISION_LABEL: Record<FinalDecision, string> = { pass: "Pass", flag: "Flag" };

const money = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });
export const formatDollars = (amount: number) => money.format(amount);

export function formatLocalTime(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

export const noteQuotes = (flags: Flag[]) =>
  flags.flatMap((f) => f.evidence.filter((e) => e.kind === "note").map((e) => e.id));

// ------------------------------------------------------------------ claim list
export function StatusBadge({ status }: { status: ReviewStatus }) {
  return <span className={`badge status-${status}`}>{STATUS_LABEL[status] ?? status}</span>;
}

export function ClaimList(props: {
  claims: ClaimSummary[] | null;
  error: string | null;
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  const { claims, error, selectedId, onSelect } = props;
  return (
    <nav className="claim-list" aria-label="Claims">
      <h2>Claims</h2>
      {error && <p className="message error" role="alert">{error}</p>}
      {!claims && !error && <p className="muted">Loading claims...</p>}
      {claims && (
        <ul>
          {claims.map((c) => (
            <li key={c.claim_id}>
              <button
                type="button"
                className={`claim-item${c.claim_id === selectedId ? " selected" : ""}`}
                aria-current={c.claim_id === selectedId ? "true" : undefined}
                onClick={() => onSelect(c.claim_id)}
              >
                <span className="claim-item-top">
                  <strong>{c.claim_id}</strong>
                  <StatusBadge status={c.review_status} />
                </span>
                <span className="claim-item-proc">
                  {c.cpt} {c.procedure ?? "(unknown procedure)"}
                </span>
                <span className="claim-item-amount">{formatDollars(c.billed_amount)}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </nav>
  );
}

// ------------------------------------------------------------------ claim card with highlighted note
export function ClaimCard({ claim, procedure, quotes }: { claim: Claim; procedure: string | null; quotes: string[] }) {
  const { segments, notFound } = highlightQuotes(claim.clinical_note, quotes);
  return (
    <section className="card" aria-labelledby="claim-heading">
      <h2 id="claim-heading">Claim {claim.claim_id}</h2>
      <dl className="facts">
        <dt>Provider</dt>
        <dd>
          {claim.provider.name} ({claim.provider.specialty})
        </dd>
        <dt>Date of service</dt>
        <dd>{claim.date_of_service}</dd>
        <dt>Procedure</dt>
        <dd>
          {claim.cpt} {procedure ?? ""}
          {claim.modifiers.length > 0 && <> (modifiers {claim.modifiers.join(", ")})</>}
          {claim.units !== 1 && <>, {claim.units} units</>}
        </dd>
        <dt>Diagnosis codes</dt>
        <dd>{claim.icd10.join(", ")}</dd>
        <dt>Billed amount</dt>
        <dd>{formatDollars(claim.billed_amount)}</dd>
      </dl>
      <h3>Clinical note</h3>
      <p className="note" data-testid="clinical-note">
        {segments.map((s, i) => (s.highlight ? <mark key={i}>{s.text}</mark> : <span key={i}>{s.text}</span>))}
      </p>
      {notFound.length > 0 && (
        <div className="message warning">
          <p>Quoted by the model but not found in the note as written:</p>
          {notFound.map((q) => (
            <blockquote key={q}>{q}</blockquote>
          ))}
        </div>
      )}
    </section>
  );
}

// ------------------------------------------------------------------ recommendation
export function RecommendationCard({ result }: { result: AgentResult }) {
  const source =
    result.explanation_source === "llm"
      ? `Explanation written by the model (${result.model}).`
      : result.explanation_source === "template"
        ? "Explanation from a fixed template, because the model's text did not pass the checks."
        : "Fixed text, no model call.";
  return (
    <section className={`card recommendation rec-${result.recommended_action}`} aria-labelledby="rec-heading">
      <h2 id="rec-heading">Agent recommendation</h2>
      <p className="rec-label">{RECOMMENDATION_LABEL[result.recommended_action] ?? result.recommended_action}</p>
      <p className="muted small">{source}</p>
      <p className="explanation">{result.explanation}</p>
      {result.errors.length > 0 && (
        <details>
          <summary>Agent errors ({result.errors.length})</summary>
          <ul>
            {result.errors.map((e, i) => (
              <li key={i}>{e}</li>
            ))}
          </ul>
        </details>
      )}
    </section>
  );
}

// ------------------------------------------------------------------ flags and evidence
export type CriterionState = { status: "loading" } | { status: "ok"; criterion: Criterion } | { status: "error"; message: string };

function PolicyChip(props: { id: string; state: CriterionState | undefined; onOpen: (id: string) => void }) {
  const [open, setOpen] = useState(false);
  const { id, state, onOpen } = props;
  const panelId = `crit-${id}-${Math.abs(hashCode(id))}`;
  return (
    <span className="chip-wrap">
      <button
        type="button"
        className="chip chip-policy"
        aria-expanded={open}
        aria-controls={panelId}
        onClick={() => {
          if (!open) onOpen(id);
          setOpen(!open);
        }}
      >
        {id}
      </button>
      {open && (
        <span className="criterion" id={panelId} role="region" aria-label={`Criterion ${id}`}>
          {(!state || state.status === "loading") && <span className="muted">Loading criterion text...</span>}
          {state?.status === "error" && <span className="error">{state.message}</span>}
          {state?.status === "ok" && (
            <>
              <strong>
                {state.criterion.criterion_id}, {state.criterion.policy_title}
              </strong>
              <span className="criterion-text">{state.criterion.text}</span>
            </>
          )}
        </span>
      )}
    </span>
  );
}

function hashCode(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0;
  return h;
}

function EvidenceList(props: {
  flag: Flag;
  criteria: Record<string, CriterionState>;
  onOpenCriterion: (id: string) => void;
}) {
  const { flag, criteria, onOpenCriterion } = props;
  const shown = flag.evidence.filter((e: Evidence) => !(e.kind === "flag" && e.id === flag.flag_id));
  const quotes = shown.filter((e) => e.kind === "note");
  const chips = shown.filter((e) => e.kind !== "note");
  return (
    <div className="evidence">
      {chips.length > 0 && (
        <div className="chips" aria-label="Evidence">
          {chips.map((e, i) =>
            e.kind === "policy" ? (
              <PolicyChip key={i} id={e.id} state={criteria[e.id]} onOpen={onOpenCriterion} />
            ) : (
              <span key={i} className="chip">
                {e.kind === "code" ? `Code ${e.id}` : `${e.kind}: ${e.id}`}
              </span>
            ),
          )}
        </div>
      )}
      {quotes.map((q, i) => (
        <blockquote key={i} className="quote">
          <span className="muted small">From the clinical note: </span>"{q.id}"
        </blockquote>
      ))}
    </div>
  );
}

export function FlagsList(props: {
  flags: Flag[];
  criteria: Record<string, CriterionState>;
  onOpenCriterion: (id: string) => void;
}) {
  const { flags, criteria, onOpenCriterion } = props;
  return (
    <section className="card" aria-labelledby="flags-heading">
      <h2 id="flags-heading">Flags ({flags.length})</h2>
      {flags.length === 0 && <p className="muted">No flags.</p>}
      <ul className="flags">
        {flags.map((f) => (
          <li key={f.flag_id} className="flag">
            <div className="flag-top">
              <code className="flag-id">{f.flag_id}</code>
              <span className={`badge source-${f.source}`}>{f.source === "rule" ? "Rule" : "Policy reading"}</span>
            </div>
            <p>{f.detail}</p>
            <EvidenceList flag={f} criteria={criteria} onOpenCriterion={onOpenCriterion} />
          </li>
        ))}
      </ul>
    </section>
  );
}

export function RejectedList({ rejected }: { rejected: AgentResult["rejected"] }) {
  if (!rejected || rejected.length === 0) return null;
  return (
    <section className="card">
      <details>
        <summary>Rejected by checks ({rejected.length})</summary>
        <ul className="rejected">
          {rejected.map((r, i) => {
            const fid = typeof r.flag?.flag_id === "string" ? r.flag.flag_id : "(no flag id)";
            const cid = typeof r.flag?.criterion_id === "string" ? r.flag.criterion_id : null;
            return (
              <li key={i}>
                <code>{fid}</code>
                {cid && <> citing <code>{cid}</code></>}: {r.reason}
              </li>
            );
          })}
        </ul>
      </details>
    </section>
  );
}

// ------------------------------------------------------------------ trace timeline
function attemptsOf(entry: TraceEntry): number {
  const a = entry.data?.attempts;
  return Array.isArray(a) ? a.length : 0;
}

export function TraceTimeline({ trace }: { trace: TraceEntry[] }) {
  return (
    <section className="card" aria-labelledby="trace-heading">
      <h2 id="trace-heading">Reasoning trace</h2>
      <ol className="timeline" aria-label="Reasoning trace">
        {trace.map((t, i) => {
          const attempts = attemptsOf(t);
          const retried = (t.node === "judge" || t.node === "explain") && attempts > 1;
          return (
            <li key={i} className={`timeline-item${retried ? " retried" : ""}`} data-node={t.node}>
              <details>
                <summary>
                  <span className="node-name">{t.node}</span> <span className="node-summary">{t.summary}</span>
                  {retried && <span className="badge retry">Retried ({attempts} attempts)</span>}
                </summary>
                <pre className="json">{JSON.stringify(t.data, null, 2)}</pre>
              </details>
            </li>
          );
        })}
      </ol>
    </section>
  );
}

// ------------------------------------------------------------------ review panel
const REVIEWER_KEY = "claims-review.reviewer";

function readReviewer(): string {
  try {
    return window.localStorage.getItem(REVIEWER_KEY) ?? "reviewer";
  } catch {
    return "reviewer";
  }
}

function saveReviewer(name: string) {
  try {
    window.localStorage.setItem(REVIEWER_KEY, name);
  } catch {
    // storage blocked; the name is still used for this submit
  }
}

export function ReviewPanel(props: {
  recommendation: RecommendedAction;
  onSubmit: (body: ActionBody) => Promise<void>;
}) {
  const { recommendation, onSubmit } = props;
  const approveBlocked = recommendation === "needs_human_review";
  const [reviewer, setReviewer] = useState(readReviewer);
  const [action, setAction] = useState<ActionKind>(approveBlocked ? "override" : "approve");
  const [decision, setDecision] = useState<FinalDecision | "">("");
  const [reason, setReason] = useState("");
  const [question, setQuestion] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [problems, setProblems] = useState<string[] | null>(null);
  const [failure, setFailure] = useState<string | null>(null);

  const hints: string[] = [];
  if (!reviewer.trim()) hints.push("A reviewer name is required.");
  if (action === "override" && !decision) hints.push("Choose a decision.");
  if (action === "override" && !reason.trim()) hints.push("A reason is required.");
  if (action === "request_info" && !question.trim()) hints.push("A question is required.");

  async function submit(e: FormEvent) {
    e.preventDefault();
    setProblems(null);
    setFailure(null);
    const body: ActionBody = { action, reviewer };
    if (action === "override") {
      if (decision) body.final_decision = decision;
      body.reason = reason;
    }
    if (action === "request_info") body.question = question;
    saveReviewer(reviewer);
    setSubmitting(true);
    try {
      await onSubmit(body);
    } catch (err) {
      if (err instanceof ApiError && err.status === 422 && err.problems) setProblems(err.problems);
      else if (err instanceof ApiError && err.status === 422)
        setFailure("The server rejected the request (422). Check the fields and try again.");
      else setFailure(`The action failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <section className="card review-panel" aria-labelledby="review-heading">
      <h2 id="review-heading">Human review</h2>
      <form onSubmit={submit} noValidate>
        <div className="field">
          <label htmlFor="reviewer">Reviewer name</label>
          <input id="reviewer" type="text" value={reviewer} onChange={(e) => setReviewer(e.target.value)} />
        </div>
        <fieldset>
          <legend>Action</legend>
          <label className="radio">
            <input
              type="radio"
              name="action"
              value="approve"
              checked={action === "approve"}
              disabled={approveBlocked}
              onChange={() => setAction("approve")}
            />
            Approve the recommendation
          </label>
          {approveBlocked && (
            <p className="muted small" id="approve-reason">
              Approve is not available: the agent did not reach a recommendation. Override or request more information.
            </p>
          )}
          <label className="radio">
            <input type="radio" name="action" value="override" checked={action === "override"} onChange={() => setAction("override")} />
            Override
          </label>
          <label className="radio">
            <input
              type="radio"
              name="action"
              value="request_info"
              checked={action === "request_info"}
              onChange={() => setAction("request_info")}
            />
            Request more information
          </label>
        </fieldset>

        {action === "override" && (
          <>
            <div className="field">
              <label htmlFor="decision">Decision</label>
              <select id="decision" value={decision} onChange={(e) => setDecision(e.target.value as FinalDecision | "")}>
                <option value="">Choose a decision</option>
                <option value="pass">Pass</option>
                <option value="flag">Flag</option>
              </select>
            </div>
            <div className="field">
              <label htmlFor="reason">Reason (required)</label>
              <textarea id="reason" rows={3} value={reason} aria-required="true" onChange={(e) => setReason(e.target.value)} />
            </div>
          </>
        )}
        {action === "request_info" && (
          <div className="field">
            <label htmlFor="question">Question (required)</label>
            <textarea id="question" rows={3} value={question} aria-required="true" onChange={(e) => setQuestion(e.target.value)} />
          </div>
        )}

        {hints.length > 0 && (
          <ul className="hints" aria-label="Hints">
            {hints.map((h) => (
              <li key={h}>{h}</li>
            ))}
          </ul>
        )}
        <button type="submit" className="primary" disabled={submitting}>
          {submitting ? "Submitting..." : "Submit"}
        </button>
      </form>
      {problems && (
        <div className="message error" role="alert">
          <p>The server did not accept this action:</p>
          <ul>
            {problems.map((p) => (
              <li key={p}>{p}</li>
            ))}
          </ul>
        </div>
      )}
      {failure && (
        <p className="message error" role="alert">
          {failure}
        </p>
      )}
    </section>
  );
}

// ------------------------------------------------------------------ decision record
export function DecisionRecordCard({ record }: { record: DecisionRecord }) {
  const h = record.human;
  return (
    <section className="card decision-record" aria-labelledby="record-heading">
      <h2 id="record-heading">Decision record</h2>
      <dl className="facts">
        <dt>Final status</dt>
        <dd>{FINAL_STATUS_LABEL[record.final_status] ?? record.final_status}</dd>
        <dt>Human action</dt>
        <dd>{ACTION_LABEL[h.action] ?? h.action}</dd>
        <dt>Decision</dt>
        <dd>{h.final_decision ? DECISION_LABEL[h.final_decision] : "None"}</dd>
        {h.reason && (
          <>
            <dt>Reason</dt>
            <dd>{h.reason}</dd>
          </>
        )}
        {h.question && (
          <>
            <dt>Question</dt>
            <dd>{h.question}</dd>
          </>
        )}
        <dt>Reviewer</dt>
        <dd>{h.reviewer}</dd>
        <dt>Time</dt>
        <dd>
          <time dateTime={h.at}>{formatLocalTime(h.at)}</time> <span className="muted small">(stored as UTC {h.at})</span>
        </dd>
        <dt>Agent recommendation</dt>
        <dd>{RECOMMENDATION_LABEL[record.agent_result.recommended_action] ?? record.agent_result.recommended_action}</dd>
      </dl>
    </section>
  );
}
