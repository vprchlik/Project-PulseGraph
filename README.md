# PulseGraph

A measurement-first future search engine for public entities. Forecasts the trajectory of GitHub repositories, identifies plausible future regimes (breakout, plateau, decay, spike-and-fade, steady growth), grounds trajectory changes in real events, and retrieves historical analogs.

## Architecture

```
pulsegraph/
├── data/           # Data ingestion (GH Archive, GitHub API, HN, Libraries.io)
├── forecast/       # Forecasting models (Chronos-2, baselines)
├── evaluation/     # Metrics (CRPS, MAPE, coverage), backtesting, calibration
├── graph/          # Entity graph construction and graph-derived features
├── regime/         # Regime classification, DTW clustering, analog retrieval
├── explain/        # Event attribution, confidence scoring, LLM narratives
└── api/            # FastAPI REST API
frontend/           # Next.js + Tailwind dashboard
scripts/            # Phase 0 feasibility, backfill, daily ingestion
tests/              # Test suite
```

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+ (for frontend)
- PostgreSQL 14+ (optional for production; prototype can use parquet files)
- A GitHub personal access token (for API access)
- Google Cloud credentials (only for BigQuery/GH Archive access)

### Installation

```bash
cd PulseGraph

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the package and dependencies
pip install -e ".[dev]"

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your tokens and credentials
```

### Frontend setup

```bash
cd frontend
npm install
```

## Quick Start (Synthetic Data)

Run the Phase 0 feasibility study with synthetic data (no external credentials needed):

```bash
python scripts/phase0_feasibility.py --data-source synthetic --n-repos 50 --n-eval-repos 20 --max-windows 5 --baselines-only
```

This will:
1. Generate synthetic GitHub-like time series data
2. Run four baseline forecasters (naive, linear, ETS, LightGBM)
3. Perform rolling backtests at 7, 30, and 90-day horizons
4. Produce a go/no-go report in `reports/`

To include Chronos-2 (requires GPU or patience on CPU):

```bash
python scripts/phase0_feasibility.py --data-source synthetic --n-repos 50 --n-eval-repos 20 --max-windows 5
```

## Quick Start (Real Data)

### Option A: BigQuery (recommended for historical backfill)

1. Set up a Google Cloud project and enable the BigQuery API
2. Create a service account and download the JSON key
3. Configure `.env`:
   ```
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
   BIGQUERY_PROJECT_ID=your-project-id
   ```
4. Run the backfill:
   ```bash
   python scripts/backfill_gharchive.py --top-n 1000 --tail-n 500
   ```
5. Run feasibility:
   ```bash
   python scripts/phase0_feasibility.py --data-source file --data-path data/raw/daily_signals.parquet
   ```

### Option B: GitHub API (for incremental updates)

Set `GITHUB_TOKEN` in `.env`, then:

```bash
python scripts/daily_ingest.py
```

## Running the API

```bash
# Start the FastAPI server
uvicorn pulsegraph.api.app:app --reload --host 0.0.0.0 --port 8000

# In another terminal, start the frontend
cd frontend && npm run dev
```

The dashboard will be available at `http://localhost:3000`.

API docs at `http://localhost:8000/docs`.

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | System status |
| `GET /entity/{owner}/{repo}/forecast?horizon=30` | Probabilistic forecast + regime probabilities |
| `GET /entity/{owner}/{repo}/explain` | Event attributions |
| `GET /search/regime?regime=breakout&horizon=30&min_prob=0.4` | Future regime search |
| `GET /search/analog?owner=X&repo=Y&lookback=30` | Historical analog retrieval |
| `GET /entity/{owner}/{repo}/calibration` | Confidence and calibration info |

## Running Tests

```bash
pytest tests/ -v
```

## Database Setup (Optional)

For production use with PostgreSQL:

```bash
# Create the database
createdb pulsegraph

# Run migrations
alembic upgrade head
```

Or create tables directly:

```python
from pulsegraph.data.schema import create_all_tables
create_all_tables()
```

## Project Phases

- **Phase 0**: Feasibility validation (synthetic + real data benchmarks)
- **Phase 1**: Single-signal forecasting pipeline
- **Phase 2**: Event grounding (releases, HN mentions, event covariates)
- **Phase 3**: Graph context (dependency/contributor graphs, graph features)
- **Phase 4**: Future regime search (regime clustering, analog retrieval, API, dashboard)
- **Phase 5**: Explanation, calibration, and polish
