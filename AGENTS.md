# AGENTS.md

## Cursor Cloud specific instructions

PulseGraph bundles two deliverables in one repo:

- **The study (primary):** an offline Python forecasting pipeline validated with `pytest`. See `README.md` / `RESULTS.md`. No long-running services are needed.
- **The web app (legacy but runnable):** a FastAPI backend (`pulsegraph/api/app.py`) plus a Next.js frontend (`frontend/`). The README calls `pulsegraph/api`, `pulsegraph/graph`, and `frontend/` legacy, but they form a complete, runnable product.

### Environment / setup

- Python deps install into a local `.venv` via `pip install -e ".[dev]"` (handled by the startup update script). Always `source .venv/bin/activate` before running Python commands. `python3-venv` is a system package the VM needs; it is already present in the snapshot.
- Frontend deps: `npm install` in `frontend/` (has a committed `package-lock.json`).

### Standard commands

- Tests: `pytest -q` (see `pyproject.toml` `[tool.pytest.ini_options]`).
- Lint (Python): `ruff check .` — the repo currently has pre-existing ruff findings (e.g. unused imports); they are not from environment setup.
- Type check: `mypy pulsegraph` — reports pre-existing missing-stub errors; informational only.
- Backend: `uvicorn pulsegraph.api.app:app --host 0.0.0.0 --port 8000`.
- Frontend: `npm run dev` in `frontend/` (serves on `:3000`, dev mode).

### Non-obvious gotchas

- **`npm run lint` is NOT configured.** It launches an interactive "How would you like to configure ESLint?" prompt (no eslint config is committed), which will hang a non-interactive shell. Do not rely on it in automation; use `npm run build` to type/compile-check the frontend instead.
- **The web app needs data that is gitignored.** The backend loads `data/raw/daily_signals.parquet` (+ optional `data/raw/target_repos.parquet`) at startup. Without them the API still starts but reports `0 repos loaded` and every forecast/search endpoint returns 404. A fresh clone has no data. Generate synthetic data (no GitHub token needed) with:
  ```python
  from pulsegraph.config import RAW_DIR
  from pulsegraph.data.loader import generate_synthetic_data
  generate_synthetic_data(n_repos=40, n_days=730, seed=42).to_parquet(RAW_DIR/"daily_signals.parquet", index=False)
  ```
  Note `scripts/phase0_feasibility.py --data-source synthetic` writes `synthetic_daily_signals.parquet`, which the API does NOT read — the API only reads `daily_signals.parquet`.
- **Chronos-2 weights download lazily from HuggingFace on the first `/forecast` call.** With network access the API reports `model_loaded: true` and forecasts are labeled `chronos2`. Without network/GPU it automatically falls back to an ETS baseline (`model_loaded: false`) — both paths are expected and non-blocking.
- **Frontend → backend wiring:** `frontend/next.config.js` rewrites `/api/:path*` to `http://localhost:8000/:path*`, so the backend must run on port 8000 for the UI to work.
- **PostgreSQL is unused at runtime.** `pulsegraph/data/schema.py`, `alembic/`, and `DATABASE_URL` exist but the API/scripts read/write parquet, not the DB. No database is required to run or test anything.
- A `GITHUB_TOKEN` is only needed to ingest *fresh* real data (`scripts/ingest_stars.py`); it is not required for tests or for running the app on synthetic data.

### Known pre-existing app bug (web app detail view)

- The web app's entity **detail view** (`frontend/src/components/EntityDetail.tsx`) fires `/forecast`, `/explain`, and `/search/analog` together in one `Promise.all`, so if any one fails the whole detail page shows an error and no chart renders. With the current unpinned deps (numpy 2.x / pandas 3.x), `/explain` fails: `pulsegraph/explain/attribution.py` does `rolling_std[rolling_std < 1e-6] = 1.0` on `pd.Series(...).rolling().std().values`, which is a **read-only** array in pandas 3.x → `ValueError: assignment destination is read-only`. The `/health`, `/forecast`, `/search/regime`, and `/search/analog` endpoints work; only `/explain` (and therefore the combined detail view) is broken. This is an application bug, not an environment/setup issue.
