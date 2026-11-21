# DECISIONS

## Decisions

- **D1.** Fresh repo and history, named `claims-review-agent`. The old history contains assessment files and a resume with personal contact info.
- **D2.** Keep Groq `openai/gpt-oss-120b`. Tests use a fake LLM. Eval reports spread over runs.

## Incidents

- **I1** (2026-11-21): Headline eval numbers came from the best of several runs; four runs gave four different results.
- **I2** (2026-11-21): Old `evaluate.py` silently dropped failed claims and once overwrote results with n=6.
- **I3** (2026-11-21): Old `evaluate.py` printed a hardcoded "same recall" sentence.
- **I4** (2026-11-21): README example named the wrong claim (CLM-0012 vs CLM-0011/0014).
- **I5** (2026-11-21): Report said 7.4 s latency; README and slides said 10.7 s.
- **I6** (2026-11-21): Deck called the policy "real CMS coverage policy" but the file header said it was unverified.
- **I7** (2026-11-21): Claude Code rewrote policy bullet 2 (laterality) without flagging it.
- **I8** (2026-11-21): A Word scripting loop may have closed documents without saving.
- **I9** (2026-11-21): Deck generator existed only in a temp dir, not in the repo.
- **I10** (2026-11-21): No tests existed.
- **I11** (2026-11-21): Resume PDF with real phone/email was in the old repo; GROQ key lives in `~/.zshrc`, old key revocation unconfirmed.
- **I12** (2026-11-21): Pricing/complexity rules in `rules.py` are invented, not sourced (must be labeled synthetic).
- **I13** (2026-11-21): CLM-0009 rationale says "left" knee for M23.221, unverified.
