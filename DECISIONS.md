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

## Incidents

- **I1** (2026-11-21): Headline eval numbers came from the best of several runs; four runs gave four different results.
- **I2** (2026-11-21): Old `evaluate.py` silently dropped failed claims and once overwrote results with n=6.
- **I3** (2026-11-21): Old `evaluate.py` printed a hardcoded "same recall" sentence.
- **I4** (2026-11-21): README example named the wrong claim (CLM-0012 vs CLM-0011/0014).
- **I5** (2026-11-21): Report said 7.4 s latency; README and slides said 10.7 s.
- **I6** (2026-11-21): Deck called the policy "real CMS coverage policy" but the file header said it was unverified.
- **I7** (2026-11-21): Claude Code rewrote policy bullet 2 (laterality) without flagging it. Update (2026-11-22): the policy body was rewritten into POL-001 during Phase 1, meaning preserved, now explicitly synthetic.
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
