import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError, type ActionBody, type ClaimDetail, type ClaimSummary, type Review } from "./api";
import {
  ClaimCard,
  ClaimList,
  DecisionRecordCard,
  FlagsList,
  RecommendationCard,
  RejectedList,
  ReviewPanel,
  StatusBadge,
  TraceTimeline,
  noteQuotes,
  type CriterionState,
} from "./components";

const APP_NAME = "Claims Review Agent";
const REPO_NAME = "claims-review-agent";

function hashId(): string | null {
  const raw = window.location.hash.replace(/^#/, "");
  return raw ? decodeURIComponent(raw) : null;
}

function describeError(err: unknown): string {
  if (err instanceof ApiError) return err.message;
  return err instanceof Error ? err.message : String(err);
}

function runErrorMessage(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === 503) return `The server cannot run a review: ${err.detail ?? "the API key is missing"}`;
    if (err.status === 409)
      return `A review for this claim already exists, so it was not started again. ${err.detail ?? ""}`.trim();
    if (err.status === 0) return err.message;
    return `The review failed (status ${err.status}): ${err.message}`;
  }
  return `The review failed: ${describeError(err)}`;
}

export default function App() {
  const [claims, setClaims] = useState<ClaimSummary[] | null>(null);
  const [listError, setListError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(hashId);
  const [detail, setDetail] = useState<ClaimDetail | null>(null);
  const [review, setReview] = useState<Review | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [criteria, setCriteria] = useState<Record<string, CriterionState>>({});
  const loadSeq = useRef(0);

  const refreshList = useCallback(async () => {
    try {
      setClaims(await api.listClaims());
      setListError(null);
    } catch (err) {
      setListError(`Could not load claims. ${describeError(err)}`);
    }
  }, []);

  useEffect(() => {
    void refreshList();
  }, [refreshList]);

  useEffect(() => {
    const onHash = () => setSelectedId(hashId());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    const seq = ++loadSeq.current;
    setDetail(null);
    setReview(null);
    setLoadError(null);
    setRunError(null);
    if (!selectedId) return;
    (async () => {
      try {
        const [d, r] = await Promise.all([api.getClaim(selectedId), api.getReview(selectedId)]);
        if (seq !== loadSeq.current) return;
        setDetail(d);
        setReview(r);
      } catch (err) {
        if (seq !== loadSeq.current) return;
        setLoadError(
          err instanceof ApiError && err.status === 404
            ? `Claim ${selectedId} was not found.`
            : `Could not load claim ${selectedId}. ${describeError(err)}`,
        );
      }
    })();
  }, [selectedId]);

  const select = (id: string) => {
    if (id === selectedId) return;
    window.location.hash = encodeURIComponent(id);
    setSelectedId(id);
  };

  async function runReview() {
    if (!selectedId) return;
    const id = selectedId;
    const seq = loadSeq.current;
    setRunning(true);
    setRunError(null);
    try {
      const started = await api.startReview(id);
      if (seq === loadSeq.current) setReview({ status: "awaiting_review", payload: started.payload, agent_result: started.agent_result });
    } catch (err) {
      setRunError(runErrorMessage(err));
    } finally {
      setRunning(false);
      void refreshList();
    }
  }

  async function submitAction(body: ActionBody) {
    if (!selectedId) return;
    const record = await api.submitAction(selectedId, body);
    setReview({ status: "completed", record });
    void refreshList();
  }

  const openCriterion = useCallback(
    (id: string) => {
      const current = criteria[id];
      if (current && current.status !== "error") return;
      setCriteria((c) => ({ ...c, [id]: { status: "loading" } }));
      api
        .getCriterion(id)
        .then((criterion) => setCriteria((c) => ({ ...c, [id]: { status: "ok", criterion } })))
        .catch((err) =>
          setCriteria((c) => ({ ...c, [id]: { status: "error", message: `Could not load ${id}. ${describeError(err)}` } })),
        );
    },
    [criteria],
  );

  const status = review?.status ?? detail?.review_status ?? null;
  const agentResult =
    review?.status === "awaiting_review" ? review.agent_result : review?.status === "completed" ? review.record.agent_result : null;
  const procedure = claims?.find((c) => c.claim_id === selectedId)?.procedure ?? null;

  return (
    <div className="app">
      <header className="app-header">
        <h1>{APP_NAME}</h1>
        <p className="banner" role="note">
          Synthetic data only. Not for real claims.
        </p>
      </header>

      <div className="layout">
        <ClaimList claims={claims} error={listError} selectedId={selectedId} onSelect={select} />

        <main className="main">
          {!selectedId && <p className="muted">Select a claim from the list.</p>}
          {loadError && (
            <p className="message error" role="alert">
              {loadError}
            </p>
          )}
          {detail && (
            <>
              <div className="status-line">
                Review status: {status && <StatusBadge status={status} />}
              </div>
              <ClaimCard claim={detail.claim} procedure={procedure} quotes={agentResult ? noteQuotes(agentResult.flags) : []} />

              {status === "not_started" && (
                <section className="card run">
                  <button type="button" className="primary" onClick={runReview} disabled={running}>
                    Run review
                  </button>
                  {running && (
                    <p className="running" role="status">
                      <span className="spinner" aria-hidden="true" /> The agent is reading the claim (about 10 seconds)
                    </p>
                  )}
                  {runError && (
                    <p className="message error" role="alert">
                      {runError}
                    </p>
                  )}
                </section>
              )}

              {status === "incomplete" && (
                <p className="message error" role="alert">
                  This review is stuck in an incomplete state (the run was interrupted before it finished). It cannot be
                  resumed or started again from here: the stored review must be deleted (runs/checkpoints.sqlite) and the
                  API restarted.
                </p>
              )}

              {agentResult && (
                <>
                  <RecommendationCard result={agentResult} />
                  <FlagsList flags={agentResult.flags} criteria={criteria} onOpenCriterion={openCriterion} />
                  <RejectedList rejected={agentResult.rejected} />
                  <TraceTimeline trace={agentResult.trace} />
                </>
              )}

              {review?.status === "awaiting_review" && (
                <ReviewPanel
                  key={selectedId}
                  recommendation={review.agent_result.recommended_action}
                  onSubmit={submitAction}
                />
              )}

              {review?.status === "completed" && <DecisionRecordCard record={review.record} />}
            </>
          )}
        </main>
      </div>

      <footer className="app-footer">{REPO_NAME}</footer>
    </div>
  );
}
