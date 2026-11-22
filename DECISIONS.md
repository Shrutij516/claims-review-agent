# DECISIONS

## Decisions

- **D1.** Fresh repo and history, named `claims-review-agent`. The old history contains assessment files and a resume with personal contact info.
- **D2.** Keep Groq `openai/gpt-oss-120b`. Tests use a fake LLM. Eval reports spread over runs.
- **D3.** Flags come from deterministic tools wherever possible. The LLM judges only what rules cannot see (clinical note vs policy text) and writes the explanation. Eval is reported per anomaly type so we can show where the LLM helps and where it does not.
- **D4.** Synthetic data policy: real ICD-10-CM codes (public), real CPT code numbers but descriptions are our own words (CPT text is AMA copyrighted), a few HCPCS Level II codes. All policies, price bands and unit limits are invented and labeled SYNTHETIC.
- **D5.** Eval split: 60 claims, 20 dev and 40 test. Test is frozen by SHA256 in `data/v2/TEST_SPLIT.sha256`. Never tune on test. Changing test data needs a hash update plus a new Decisions entry.
- **D6.** Policies have stable ids (POL-001) and numbered criteria (POL-001.2) so every decision can cite evidence precisely.
- **D7.** Data is produced by a seeded generator (seed 42) into plain JSON. No database.
- **D8.** v1 data and baseline stay untouched until the Phase 2 baseline replaces them.
- **D9.** Python is managed by uv (`.venv`, Python 3.12) because Homebrew python@3.12 3.12.13_2 fails at ensurepip (pyexpat symbol mismatch with macOS libexpat).
- **D10.** Phase 1 deviations accepted: code tables are objects with a note field plus a codes list; `flag_catalog.json` has a parameters block (price bands, unit limits, tiers, windows) and a synthetic_notice; HCPCS J codes sit in the claim `cpt` field; `--freeze` is the only way to write `TEST_SPLIT.sha256`; the generator re-derives every R_ flag from the catalog and fails on mismatch.
- **D11.** Retrieval is BM25 (`rank_bm25`) over one chunk per numbered policy criterion (id like POL-003.2). Deterministic, free, easy to explain, and every hit is directly citable.
- **D12.** Tools are pure typed functions with no LLM. The rule checks are implemented independently of the generator's check, so agreement on all 60 claims is a real cross-check. If they disagree, one implementation or one label is wrong: investigate and log it, never edit a label just to pass.
- **D13.** A claim whose processing fails is counted as an error, never dropped. In metrics an error counts as wrong (an error on an anomalous claim is a false negative, on a clean claim a false positive) and is reported separately.
- **D14.** Tuning protocol: tools, retrieval and prompts are tuned on dev only. The test split is run only in Phase 6, behind an explicit `--final` flag.
- **D15.** v1 files (`src/rules.py`, `src/baseline.py`, `src/generate_claims.py`, `data/claims.json`, `data/ground_truth.json`, `tests/test_baseline.py`) are retired at the end of this phase in a separate commit, once the new baseline exists.
- **D16.** Known simplification: one CPT per claim. Real claims have several lines. Listed as a README limitation.
- **D17.** (Claude Code, Phase 2) Rule flag evidence uses criterion ids from a static map in `src/claims_review/rules.py` (`policy_criteria_for`), because the catalog lists policy ids only. A test checks that every mapped criterion exists and belongs to a policy the catalog lists for that flag.
- **D18.** (Claude Code, Phase 2) Retrieval indexes the policy title together with the criterion text, lowercases alphanumeric tokens, drops a short fixed stopword list, and returns only positive-score hits. These choices were fixed before any retrieval metric was computed.
- **D19.** (Claude Code, Phase 2) A claim with an empty ICD-10 list is a parse error. If the procedure code is not in the tables, only R_INVALID_CODE is returned, because every other check depends on the procedure. The scripts use only the chosen split's claims, including as the cross-claim store, so dev runs never read test claims.
- **D20.** (Claude Code, Phase 2) `rank_bm25` and its dependency numpy were installed from the package index. That was the only network use in Phase 2.

## Incidents

- **I1** (2026-11-21): Headline eval numbers came from the best of several runs; four runs gave four different results.
- **I2** (2026-11-21): Old `evaluate.py` silently dropped failed claims and once overwrote results with n=6.
- **I3** (2026-11-21): Old `evaluate.py` printed a hardcoded "same recall" sentence.
- **I4** (2026-11-21): README example named the wrong claim (CLM-0012 vs CLM-0011/0014).
- **I5** (2026-11-21): Report said 7.4 s latency; README and slides said 10.7 s.
- **I6** (2026-11-21): Deck called the policy "real CMS coverage policy" but the file header said it was unverified.
- **I7** (2026-11-21): Claude Code rewrote policy bullet 2 (laterality) without flagging it. Update (2026-11-22): the policy body was rewritten into POL-001 during Phase 1 and is now explicitly synthetic. The earlier note that the meaning was preserved was wrong: a sentence about note-based degenerative findings was added to POL-001.1 (see I23).
- **I8** (2026-11-21): A Word scripting loop may have closed documents without saving.
- **I9** (2026-11-21): Deck generator existed only in a temp dir, not in the repo.
- **I10** (2026-11-21): No tests existed.
- **I11** (2026-11-21): Resume PDF with real phone/email was in the old repo; GROQ key lives in `~/.zshrc`, old key revocation unconfirmed. Update (2026-11-22): the Groq key was rotated and the old key deleted.
- **I12** (2026-11-21): Pricing/complexity rules in `rules.py` are invented, not sourced (must be labeled synthetic).
- **I13** (2026-11-21): CLM-0009 rationale says "left" knee for M23.221, unverified. Update (2026-11-22): v1 data is retired, and v2 laterality comes from the code table.
- **I14** (2026-11-22): Reviewer's Phase 0 verification missed the policy file, test file and src diff; fixed by a second check.
- **I15** (2026-11-22): System Python was 3.9 with LibreSSL, too old for LangGraph.
- **I16** (2026-11-22): Claude Code's Phase 0 "hits list" was the pre-edit list, not the post-edit state; verified by diff instead.
- **I17** (2026-11-22): Word recovery check: no recovery pane, AutoRecovery folder empty, nothing lost. Closed.
- **I18** (2026-11-22): Reviewer instruction mistakes during Groq key rotation: clipboard order contradicted itself, and the sed command failed on a newline. Fixed with a read -s prompt.
- **I19** (2026-11-22): Homebrew python@3.12 3.12.13_2 venv creation failed (pyexpat symbol not found). Fixed with uv (D9).
- **I20** (2026-11-22): Claude Code's Phase 1 test file contained literal dashes because the write tool decoded escapes; caught by its own scan, fixed with `chr()`.
- **I21** (2026-11-22): Claude Code used `/tmp` for a throwaway copy despite the scratchpad rule.
- **I22** (2026-11-22): The v1 JSON files contain escaped em dashes that cannot be edited; removed when v1 is retired (D15).
- **I23** (2026-11-22): Claude Code added a sentence to POL-001.1 (note-based degenerative findings) without flagging it, and the reviewer did not catch it at first. The I7 update line wrongly said "meaning preserved" and is corrected above. The sentence is kept on purpose because P_DEGENERATIVE_ONLY needs it.
- **I24** (2026-11-22): Claude Code did not log its Phase 1 deviations in DECISIONS.md; now D10.
- **I25** (2026-11-22): Reviewer's check command paged through less and stalled the paste; use `git --no-pager`.
