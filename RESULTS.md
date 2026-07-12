# RESULTS — Do time-series foundation models beat simple baselines at forecasting shock-driven GitHub star signals?

*Empirical study on real GitHub data. The primary six-model run is supplemented
by post-hoc adversarial checks for stronger baselines, scale domination,
dependent windows, cohort construction, and pretraining overlap. These checks
materially narrow the original interpretation and are reported rather than
hidden.*

---

## 1. The question

> Do time-series foundation models (**Chronos-2**, **TimesFM-2.5**) beat simple
> baselines at forecasting daily GitHub-star attention dynamics, and where is
> any advantage robust?

The signal of interest is the **daily count of new stars** a repository receives
— a noisy, often spiky, sometimes zero-inflated series. The present backtest
mostly evaluates *early post-launch and post-peak dynamics*; it does not test
whether a model can predict a previously unseen viral shock before it occurs.

---

## 2. Data

**Source.** Real per-star `starred_at` timestamps pulled from the GitHub REST
stargazers API (`scripts/ingest_stars.py`), aggregated into true daily
new-star counts. No synthetic data is used in the reported experiment.

**Outcome-conditioned discovery cohorts.** Repositories were discovered
reproducibly via three GitHub Search API queries:

| Cohort label | Discovery-time query intent |
|--------------|-----------------------------|
| **breakout-selected** | recently created and already high-starred |
| **steady-selected** | old, moderate-starred, and recently maintained |
| **dead-selected** | not pushed in years at ingestion time |

These are **discovery-time outcome labels**, not states available at each
forecast cutoff. The primary windows occur early in each repository's history.
In particular, the dead-selected cohort was usually still active in its
evaluated windows, so it must not be interpreted as a forecast-time "flat/dead"
regime.

**What was pulled.** 208 repos successfully ingested (2 dropped on transient
GitHub 500s), **414,271 daily rows**, global span **2008-11-13 → 2026-06-22**,
zero repos truncated by the 40k-star API pagination cap.

| Discovery cohort | Repos | Median total stars | Median history | Usable (≥270 days) |
|------------------|-------|--------------------|----------------|--------------------|
| breakout | 70 | 13,891 | 351 d | **41** |
| steady | 69 | 14,747 | 4,270 d | **69** |
| dead | 69 | 10,228 | 4,067 d | **69** |

29 breakout repos are too recently created (<270 days) for 90-day-horizon
backtesting and are excluded, leaving **179 usable repos**.

**Spot-check examples (verifiable on GitHub):**

- **breakout** `Tencent-Hunyuan/Hunyuan3D-2` — 14,002 stars, peak **983 stars in one day** (2025-01-22), then decay to single digits/day.
- **steady** `owasp-amass/amass` — 14,747 stars over 8 years, mean 5.2/day, peak only 33/day.
- **dead** `dvdzkwsk/react-redux-starter-kit` — 10,228 stars, early-2015 peak of 98/day, now a flat ~1/day tail.

---

## 3. Method

**Backtest.** Walk-forward rolling-origin evaluation
(`pulsegraph/evaluation/backtest.py`): expanding context starting 180 days
after the first observed star, step 30 days, up to 8 windows per repo, and
horizons **7 / 30 / 90 days**. Every model sees only
`context = values[:cutoff]`; no direct model-input look-ahead was found.

**Models.**

- Primary baselines: horizon-dependent `naive_seasonal`, `linear`, damped
  `ets`, and lag-feature `lightgbm`. All emit 200 predictive trajectories via
  residual resampling.
- Foundation models: **Chronos-2** (`amazon/chronos-2`) and **TimesFM-2.5** (`google/timesfm-2.5-200m-pytorch`), zero-shot. Both are quantile models; we reconstruct 200 pseudo-samples per horizon step by inverse-CDF interpolation of their native quantile levels (Chronos: 0.01–0.99; TimesFM: 0.1–0.9), so the evaluation metrics are computed identically for all six models.
- Post-hoc baseline sensitivity: deterministic last-value, fixed weekly
  seasonal-naive, and 28-day local-mean models on the exact same windows. These
  are kept in a separate artifact rather than silently folded into the primary
  run.

**Metrics.** Empirical **CRPS**, **MAE**, and supported interval coverage.
All models support central 80% intervals. TimesFM exposes only q0.1–q0.9, so a
95% interval is **not identifiable** and is no longer reported for TimesFM.
The archived primary run used ordinary empirical PIT values; those are not used
for calibration claims because non-randomized PIT is invalid for tied,
discrete, zero-inflated outcomes. The implementation now uses randomized PIT
for future runs.

**Validation of the headline result** (`scripts/validate_chronos.py`,
`scripts/test_crps_fairness.py`):

1. **No direct model-input leakage** — harness inspection confirms models never
   see target values after the cutoff.
2. **Repository-clustered inference** — windows overlap, so inference first
   averages within repository and then compares repositories, with bootstrap
   confidence intervals and Benjamini-Hochberg correction.
3. **Scale sensitivity** — results are recomputed as equal-weight
   per-repository averages and skill relative to naive forecasts.
4. **Stronger baselines** — the local-mean sensitivity model is substantially
   stronger than every original baseline.
5. **Sampling representation** — a Normal control finds only a −0.42% average
   CRPS shift between deterministic quantile grids and random samples. This is
   reassuring but not a complete proof for clipped discrete forecasts.

---

## 4. Results

### 4.1 Primary-run CRPS by discovery cohort × horizon (lower = better)

| Discovery cohort | H | naive | linear | ets | lightgbm | **chronos2** | **timesfm** |
|------------------|---|-------|--------|-----|----------|----------|---------|
| breakout | 7  | 11.82 | 11.19 | 12.59 | 14.82 | 6.43 | **6.23** |
| breakout | 30 | 13.32 | 16.69 | 18.11 | 19.40 | 7.90 | **7.68** |
| breakout | 90 | 14.55 | 28.00 | 23.99 | 23.33 | 8.52 | **8.20** |
| steady | 7  | 2.50 | 2.41 | 2.11 | 4.39 | 1.46 | **1.44** |
| steady | 30 | 3.30 | 3.99 | 2.83 | 5.64 | 2.04 | **2.02** |
| steady | 90 | 3.29 | 6.41 | 3.07 | 6.15 | **2.21** | 2.22 |
| dead | 7  | 3.12 | 2.82 | 3.02 | 5.10 | 2.43 | **2.35** |
| dead | 30 | 3.96 | 3.66 | 3.42 | 6.21 | 2.47 | **2.39** |
| dead | 90 | 4.10 | 5.68 | 3.75 | 7.46 | 2.73 | **2.67** |

Against the four original baselines, both foundation models have lower mean
CRPS in every cell. This primary table is numerically correct, but the baseline
set was not strong enough for the claim to stop here.

### 4.2 Stronger-baseline sensitivity

A 28-day local mean is much stronger than the original four baselines:

| Cohort | H | local mean | chronos2 | timesfm |
|--------|---|------------|----------|---------|
| breakout-selected | 7 | 7.77 | 6.43 | **6.23** |
| breakout-selected | 30 | 8.77 | 7.90 | **7.68** |
| breakout-selected | 90 | 9.41 | 8.52 | **8.20** |
| steady-selected | 7 | 1.66 | 1.46 | **1.44** |
| steady-selected | 30 | 2.26 | 2.04 | **2.02** |
| steady-selected | 90 | 2.42 | **2.21** | 2.22 |
| dead-selected | 7 | 2.75 | 2.43 | **2.35** |
| dead-selected | 30 | 2.64 | 2.47 | **2.39** |
| dead-selected | 90 | 2.85 | 2.73 | **2.67** |

The foundation models retain lower **mean** CRPS in all nine cells. However,
repository-level paired inference gives a more qualified result:

- **breakout-selected:** TimesFM is significantly better than local mean at all
  horizons; Chronos-2 at 7/30d but not 90d.
- **dead-selected:** TimesFM is significantly better at all horizons;
  Chronos-2 at 7/30d but not 90d.
- **steady-selected:** neither foundation model is significantly better at any
  horizon after repository clustering and multiplicity correction.

Repository win rates versus local mean range from **40.6% to 80.5%**. Mean gains
can therefore coexist with a near-zero median difference: the foundation models
mainly avoid a minority of large misses rather than winning on every typical
repository. See `reports/review_cluster_inference.csv` and
`reports/figures/crps_sensitivity_by_cohort.png`.

This mean-CRPS conclusion also survives local-mean lookbacks of 7, 14, 28, 56,
and 90 days; no single lookback beats a foundation model's cell mean. Lookback
selection was post hoc, so this is a sensitivity grid rather than tuned-model
evidence (`reports/review_local_mean_lookbacks.parquet`).

### 4.3 MAE (point accuracy)

| Regime | H | naive | linear | ets | lightgbm | chronos2 | timesfm |
|--------|---|-------|--------|-----|----------|----------|---------|
| breakout | 7  | 14.04 | 14.49 | 14.44 | 15.56 | 8.29 | **8.00** |
| breakout | 90 | 17.25 | 31.51 | 26.17 | 24.10 | 10.20 | **9.78** |
| steady | 7  | 3.02 | 3.04 | 2.60 | 4.59 | 1.82 | **1.80** |
| dead | 90 | 5.00 | 6.57 | 4.59 | 7.75 | 3.26 | **3.25** |

Foundation models have lower mean MAE in these selected cells. As with CRPS,
the improvement is skewed: avoiding catastrophic errors is more defensible than
claiming uniform point-forecast superiority.

### 4.4 Calibration

At the mutually supported **80%** level, TimesFM is closest overall to nominal
coverage (**0.774–0.852**); Chronos-2 covers **0.729–0.784**. The local mean
sensitivity baseline covers **0.688–0.885**. LightGBM covers only
**0.133–0.257**, as expected from in-sample residuals around one deterministic
recursive path; this is a weak uncertainty construction, not evidence that
LightGBM cannot be calibrated.

Chronos-2's supported 95% interval covers 0.953–0.968. No comparable TimesFM
95% interval exists in the installed API, so the earlier TimesFM 95%-coverage
claim has been withdrawn. Archived ordinary PIT plots have also been withdrawn
because deterministic PIT is not valid for discrete tied counts. See
`reports/figures/coverage80_by_regime.png`.

### 4.5 Figures

- `reports/figures/crps_by_regime.png` — primary-run mean CRPS by discovery
  cohort/horizon/model.
- `reports/figures/crps_sensitivity_by_cohort.png` — foundation models versus
  the stronger local-mean baseline.
- `reports/figures/coverage80_by_regime.png` — supported 80% interval coverage.
- `reports/figures/spotcheck_forecasts.png` — trailing-window visual checks;
  these are explicitly separate from the primary early-life backtest.

---

## 5. What the cohort breakdown does—and does not—show

- **Breakout-selected:** 78% of repositories reach their global peak before the
  evaluated span; only 12.2% peak inside it. The positive result is therefore
  mainly about **post-breakout decay**, not predicting shock onset.
- **Steady-selected:** foundation models lower mean error, but the typical
  repository is not significantly improved relative to local mean.
- **Dead-selected:** the evaluated windows have a median 5.02 stars/day and only
  7.7% zero days. They represent those repositories' early lifecycle, not their
  later dead/flat state.

Thus this is a discovery-cohort breakdown, not a valid forecast-time regime
classifier. A future study must place windows near ingestion time or assign
regimes using context-only features at each cutoff.

---

## 6. Plain-language verdict

**Qualified positive on mean robustness; not yet a general shock-forecasting
result.**

- Both foundation models have lower mean CRPS than the four primary baselines
  and the post-hoc 28-day local mean in every discovery-cohort/horizon cell.
- The strongest repository-level evidence is for the breakout-selected cohort
  at 7/30d (and TimesFM at 90d), plus the dead-selected cohort at 7/30d.
- No statistically reliable typical-repository advantage remains in the
  steady-selected cohort after adding local mean and clustering by repository.
- The gain is principally **robustness against large misses**, not uniform
  superiority.
- TimesFM has marginally lower central error and well-calibrated supported 80%
  intervals. Chronos supports wider tails, but a direct 95% comparison with
  TimesFM is impossible.

The defensible current conclusion is narrower than the original one: general
time-series foundation models are promising robust forecasters for these
selected early-life/post-peak GitHub-star trajectories, but this experiment
does not show that they predict future viral shocks or forecast-time
dead/steady regimes.

## 7. Limitations that matter for interpretation

- **Outcome-conditioned cohorts and early windows.** Labels use information
  available years after many forecast cutoffs.
- **No guaranteed temporal holdout from pretraining.** No disclosed corpus
  contains GitHub stars, but complete item-level training cutoffs are not
  independently auditable. See `reports/pretraining_overlap_audit.md`.
- **Overlapping windows.** Primary per-window p-values were anti-conservative;
  conclusions above use repository-clustered sensitivity inference instead.
- **Post-hoc stronger baselines.** Local mean was added after the primary result
  was seen; it is a transparent sensitivity analysis, not a preregistered model.
- **Primary baseline randomness.** Original residual resampling was unseeded.
  The code is now deterministic, but the archived primary artifact cannot be
  regenerated bit-for-bit.
- **Observation endpoints.** Series start at first star and end at last star.
  Trailing censoring is usually small (median 0–3 days; maximum 42 days), but a
  future trailing-window study should explicitly extend every series to the
  ingestion snapshot.
- **Current-stargazer survivor bias.** GitHub's endpoint returns stars that
  still exist at ingestion time; historical stars later removed by users are
  absent, so old daily counts are not a complete event ledger.
- **Single platform/signal.** External validity beyond GitHub stars is unknown.

*Primary pipeline: `scripts/ingest_stars.py` →
`scripts/run_experiment.py --include-chronos` → `scripts/add_timesfm.py`.
Audit: `scripts/review_integrity.py`, `scripts/review_windows.py`,
`scripts/review_scale.py`, `scripts/review_strong_baselines.py`,
`scripts/review_cluster_inference.py`, and `scripts/review_censoring.py`.
Figures: `scripts/final_analysis.py`, `scripts/generate_spotcheck.py`.*
