# PulseGraph

PulseGraph is an **empirical forecasting study**, not a search engine, agent, or
frontend product. It asks whether zero-shot time-series foundation models improve
probabilistic forecasts of **daily new GitHub-star counts** compared with simple
baselines. Real stargazer histories from 208 ingested repositories (179 usable
after history filters) are grouped into discovery-time cohorts—41
breakout-selected, 69 steady-selected, 69 dead-selected—and backtested at 7-, 30-,
and 90-day horizons against naive seasonal, linear, ETS, LightGBM, Chronos-2,
TimesFM-2.5, and stronger baselines added during audit (last-value, weekly naive,
28-day local mean).

## Research question

> Do time-series foundation models (**Chronos-2**, **TimesFM-2.5**) beat simple
> baselines at forecasting shock-driven attention signals (daily GitHub new-star
> counts), at 7/30/90-day horizons, and for which series types?

## Headline finding

Both foundation models retain lower **mean CRPS** than the original baselines and
the post-audit 28-day local-mean baseline in every discovery-cohort/horizon cell.
After repository-clustered inference and multiplicity correction, gains narrow
substantially—especially in the steady-selected cohort. The defensible
interpretation is **robustness against a minority of large misses**, not uniform
superiority.

This study does **not** demonstrate prediction of viral shock onset. Cohort
labels (breakout/steady/dead-selected) use discovery-time queries and later
outcomes; they are not forecast-time regime identifiers.

Full tables, limitations, and audit methods:

- [Results and limitations](RESULTS.md)
- [Technical write-up](WRITEUP.md)
- [Study audit](STUDY_AUDIT.md)

## Quick start

Requires Python 3.10+. A GitHub token is needed only if you collect a fresh data
snapshot; archived artifacts under `reports/` reproduce the published numbers.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

pytest -q
```

Optional: ingest new data and rerun the experiment pipeline.

```bash
# Set GITHUB_TOKEN (see .env.example), then collect stargazer histories.
python scripts/ingest_stars.py --per-regime 80

# Primary baselines + Chronos-2.
python scripts/run_experiment.py --include-chronos --tag real_full

# Add TimesFM-2.5 on the same windows.
python scripts/add_timesfm.py

# Stronger-baseline sensitivity check from the audit.
python scripts/review_strong_baselines.py
```

Or run `./setup.sh` for venv creation and dependency install.

Additional audit scripts in `scripts/review_*.py` check run integrity, window
placement, scale sensitivity, clustered inference, and endpoint censoring against
artifacts in `reports/`.

## Project structure

In-scope code for the study lives under `pulsegraph/` and `scripts/`:

```text
data/                  Raw and processed star-count series (gitignored snapshots)
pulsegraph/
  data/                GitHub API ingestion and series loading
  forecast/            Baselines, Chronos-2, TimesFM-2.5 forecasters
  evaluation/          Rolling-origin backtest and CRPS/MAE/coverage metrics
scripts/
  ingest_stars.py      Collect per-star timestamps from GitHub
  run_experiment.py    Run primary backtest pipeline
  add_timesfm.py       Add TimesFM-2.5 to existing experiment windows
  review_*.py          Post-hoc audit and sensitivity scripts
reports/               Stored experiment outputs, summaries, and figures
tests/                 Test suite
RESULTS.md             Full results tables and interpretation
WRITEUP.md             Draft technical report
STUDY_AUDIT.md         Audit methods, findings, and scope limits
```

Legacy modules (`pulsegraph/api`, `pulsegraph/graph`, `pulsegraph/regime`,
`frontend/`, and related dependencies) remain in the tree from earlier
prototypes but are **out of scope** for this study and not required to reproduce
it. Kronos exploration lives on a separate branch/worktree (`explore/kronos`).

## Reproducibility notes

- The archived study artifacts are internally audited, but a fresh GitHub API pull
  is not bit-for-bit identical: Search API results and star histories change over
  time, and primary baseline trajectory draws were unseeded.
- Core entry points: `scripts/ingest_stars.py` (data), `scripts/run_experiment.py`
  (primary run), `scripts/review_strong_baselines.py` (audit sensitivity).
- Contributions welcome via issues or PRs that preserve the study's honest scope
  boundaries; extend tests when changing forecasting or evaluation logic.

## Status

The completed study is suitable as a cautious empirical note or blog post. It is
not a released paper, DOI-backed dataset, or prospective benchmark. A stronger
follow-up would freeze a protocol, use context-only cohort features, collect a
post-model-release holdout, and preregister stronger count-aware baselines.
