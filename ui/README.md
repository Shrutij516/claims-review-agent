# Claims Review Agent UI (synthetic data only, dev split)
1. Run the API from the repo root: `.venv/bin/uvicorn --factory --app-dir src claims_review.api:create_app` (needs GROQ_API_KEY only to run reviews).
2. Run the UI: `cd ui && npm install && npm run dev`, then open http://localhost:5173 (set VITE_API_URL if the API is not on http://localhost:8000).
3. Test: `npm test` (Vitest with a mocked fetch, no network) and `npm run typecheck`.
4. Build: `npm run build` (output in ui/dist).
