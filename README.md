# PulseGraph

**PulseGraph is an empirical forecasting study** that tests whether zero-shot time-series foundation models improve probabilistic forecasts of **daily new GitHub star counts** compared with simple baselines on real repository data.

---

## Why this matters

Attention signals—GitHub stars, package downloads, social mentions—often spike at launch, decay unevenly, and sit on long flat tails. Most public time-series benchmarks focus on smooth demand or energy series; bursty, zero-inflated count data is underrepresented. This repo provides a reproducible backtest on **414k daily observations** from **179 repositories**, comparing **Chronos-2** and **TimesFM-2.5** against naive seasonal, linear, ETS, LightGBM, and stronger count-aware baselines.

---

## Key findings

| | |
|---|---|
| **Question** | Do Chronos-2 and TimesFM-2.5 beat simple baselines at forecasting daily new-star counts at 7-, 30-, and 90-day horizons? |
| **Data** | 179 usable repos (from 208 ingested), three **discovery-time cohorts** (breakout / steady / dead-selected), rolling-origin backtest |
| **Metrics** | CRPS (primary), MAE, 80% interval coverage |
| **Headline** | Both foundation models achieve **lower mean CRPS** than primary and stronger baselines in every cohort × horizon cell |
| **Where it helps most** | Longer horizons and bursty (breakout-selected) series; gains are mainly **robustness against large misses**, not uniform superiority |
| **What this is not** | Evidence that models **predict viral shock onset**—cohort labels reflect discovery-time outcomes, not forecast-time regimes |

For full tables, calibration details, sensitivity checks, and limitations, see **[RESULTS.md](RESULTS.md)**.

**Figures** (archived in `reports/figures/`):

| Figure | Description |
|--------|-------------|
| [crps_by_regime.png](reports/figures/crps_by_regime.png) | Mean CRPS by cohort, horizon, and model |
| [crps_sensitivity_by_cohort.png](reports/figures/crps_sensitivity_by_cohort.png) | Foundation models vs. stronger local-mean baseline |
| [coverage80_by_regime.png](reports/figures/coverage80_by_regime.png) | 80% interval coverage |
| [spotcheck_forecasts.png](reports/figures/spotcheck_forecasts.png) | Visual spot-checks (separate from primary backtest) |

**Further reading:** [WRITEUP.md](WRITEUP.md) (technical report draft) · [STUDY_AUDIT.md](STUDY_AUDIT.md) (methods audit and scope limits)

---

## Reproduce

Requires **Python 3.10+**. A GitHub token is only needed to collect fresh data; stored artifacts under `reports/` reproduce the published numbers without re-ingesting.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

Optional — collect data and rerun the experiment:

```bash
export GITHUB_TOKEN=...   # see .env.example
python scripts/ingest_stars.py --per-regime 80
python scripts/run_experiment.py --include-chronos --tag real_full
python scripts/add_timesfm.py
```

Or run `./setup.sh` for venv setup, dependency install, and directory scaffolding.

**Note:** A fresh GitHub API pull will not be bit-for-bit identical to the archived run (search results and star histories change over time).

---

## Project structure

Study code lives under `pulsegraph/` and `scripts/`:

```text
pulsegraph/
  data/           GitHub API ingestion and series loading
  forecast/       Baselines, Chronos-2, TimesFM-2.5
  evaluation/     Rolling-origin backtest; CRPS, MAE, coverage
scripts/
  ingest_stars.py       Collect per-star timestamps from GitHub
  run_experiment.py     Primary backtest pipeline
  add_timesfm.py        Add TimesFM-2.5 to existing experiment windows
  review_*.py           Post-hoc sensitivity and audit scripts
reports/              Experiment outputs, summaries, and figures
tests/                Test suite
RESULTS.md            Full results and interpretation
WRITEUP.md            Technical report draft
STUDY_AUDIT.md        Audit methods and scope limits
```

---

## Repository note

Earlier prototypes left legacy modules in the tree (`pulsegraph/api`, `pulsegraph/graph`, `frontend/`, and related dependencies). They are **not part of this study** and are not required to reproduce the forecasting experiment.

---

## Citation

If you use this data or methodology, please link to this repository and cite the models evaluated: [Chronos-2](https://huggingface.co/amazon/chronos-2) (Amazon) and [TimesFM-2.5](https://huggingface.co/google/timesfm-2.5-200m-pytorch) (Google). This is an empirical study, not a released paper or DOI-backed dataset.

---

## Contributing

Issues and pull requests are welcome. Please preserve the study's scope boundaries and extend tests when changing forecasting or evaluation logic.
