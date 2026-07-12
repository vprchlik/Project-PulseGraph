# PulseGraph study audit

Date: 2026-07-12  
Scope: forecasting study only; legacy API/graph/frontend code and the isolated
Kronos worktree were not touched.

## Executive verdict

No code bug was found that invalidates the stored aggregate six-model
comparison. The raw artifact is internally consistent: all six models share the
same 4,122 repository/horizon/cutoff keys, there are no duplicate rows, and all
summary fields recompute from the parquet.

The **original scientific interpretation was nevertheless too broad**. The
current design mainly evaluates early-life and post-peak attention trajectories
in cohorts selected using later outcomes. It does not establish that foundation
models predict viral shock onset, nor that they forecast series that were
dead/steady at forecast time.

After adding a substantially stronger 28-day local-mean baseline, both
foundation models still have lower **mean** CRPS in every cohort/horizon cell.
Repository-clustered inference is more qualified: reliable typical-repository
gains remain in the breakout-selected cohort and parts of the dead-selected
cohort, while no reliable gain remains in the steady-selected cohort. The
positive result is best described as **robustness against a minority of large
misses**, not universal superiority.

## Severity-ranked findings

### Invalidating for specific original claims

1. **Forecast-time regime claims were invalid.**
   - Cohorts are assigned at ingestion using later success, maintenance, or
     inactivity.
   - Backtests start 180 days after the first observed star and stop after at
     most eight monthly steps.
   - Dead-selected evaluation windows have a median 5.02 stars/day and only
     7.7% zero days. They are not generally dead or flat.
   - Action: `RESULTS.md` and `WRITEUP.md` now use
     *breakout/steady/dead-selected discovery cohort* and explicitly reject a
     forecast-time regime interpretation.

2. **Shock-prediction claims were invalid.**
   - The global peak occurs before evaluation for 78.0% of breakout-selected
     repositories; only 12.2% peak within the evaluated span.
   - Action: final materials now describe the task as forecasting early-life
     and mostly post-breakout decay. They do not claim to predict unseen viral
     shocks.

No confirmed issue invalidated the narrower aggregate statement that foundation
models reduce mean CRPS on this selected sample and window policy.

### Must-fix issues addressed

1. **The original baseline set supported a strawman attack.**
   - The original `naive_seasonal` changes period with horizon, ETS has no
     weekly seasonality, and no last-value/local-level baseline was present.
   - Added deterministic last-value, fixed weekly seasonal-naive, and 28-day
     local-mean forecasters.
   - Evaluated them on identical windows without rerunning foundation models.
   - Local mean is much stronger than every original baseline. Foundation
     models retain lower cell-mean CRPS, but repository win rates fall to
     40.6–80.5%.
   - A 7/14/28/56/90-day lookback grid confirms that no tested local-mean
     lookback beats a foundation model's cell mean.

2. **Overlapping-window inference was anti-conservative.**
   - The old Wilcoxon analysis treated repeated, overlapping windows as
     independent; 90-day targets overlap at a 30-day step.
   - Replaced inferential claims with repository-level paired comparisons:
     average windows within repository, bootstrap repositories, perform paired
     Wilcoxon tests across repositories, and control false discovery rate over
     18 comparisons.
   - TimesFM is significant versus local mean in all breakout-selected and
     dead-selected cells. Chronos is significant at 7/30d in those cohorts, not
     90d. Neither model is significant in the steady-selected cohort.

3. **Scale domination was untested.**
   - Added equal-repository macro CRPS, per-repository skill, leave-one-repo-out
     checks, and contribution concentration.
   - Macro rankings preserve a foundation-model winner in every cell.
   - Removing any single repository never changes a foundation-model winner
     into a baseline winner.
   - Gains are still concentrated: the five largest positive contributors
     account for roughly 30–77% of positive gain depending on cell. This is
     disclosed rather than hidden.

4. **TimesFM's 95% interval was mislabeled.**
   - The installed API returns q0.1–q0.9 only. Clamping outside that range makes
     the historical "95%" interval effectively q0.1–q0.9.
   - The wrapper now emits NaN for unsupported 95% bounds and records supported
     interval levels in metadata.
   - All TimesFM 95%-coverage and tail-ranking claims were removed. Final
     calibration comparisons use the mutually supported 80% interval.

5. **Ordinary PIT was invalid for discrete tied counts.**
   - Archived `mean(samples <= actual)` PIT values are not uniform under a
     calibrated discrete forecast.
   - Future metrics now use randomized empirical PIT,
     `F(y-) + U(F(y)-F(y-))`.
   - The misleading PIT figure and all archived PIT-uniformity claims were
     removed. A supported 80%-coverage figure replaces it.

6. **Pretraining contamination was unresolved.**
   - Official model cards and the Chronos-2 technical report list no GitHub-star
     dataset. TimesFM's disclosed web-attention sources end in 2022/Nov 2023;
     Chronos lists Wiki pageviews but no GitHub entities.
   - No evidence of direct entity-level memorization was found.
   - Complete item-level cutoffs are not independently auditable, and no
     270-day-eligible repository was created after either model release.
   - Action: final materials call this a novel dataset with no *known* direct
     overlap, not a guaranteed temporal holdout. Full sources and reasoning are
     in `reports/pretraining_overlap_audit.md`.

7. **Primary probabilistic baselines were not exactly reproducible.**
   - Stored bootstrap draws were unseeded.
   - Baseline code now derives stable per-model/repository/window RNG seeds.
   - The archived primary artifact remains internally valid but cannot be
     regenerated bit-for-bit. Its metadata now states this.

8. **LightGBM calibration was overinterpreted.**
   - Its uncertainty uses in-sample residuals sampled independently around one
     deterministic recursively generated path, omitting model and recursive
     trajectory uncertainty.
   - Final text now calls this a weak uncertainty construction, not evidence
     that LightGBM itself is intrinsically uncalibratable.

9. **Observation endpoints were implicit.**
   - The loader spans first through last observed star. Internal zero filling is
     correct for complete event days.
   - Relative to the ingestion snapshot, trailing gaps have medians of 0–3 days,
     95th percentiles of 1–19 days, and maximum 42 days. This does not affect
     the current early-window results materially, but would affect a redesigned
     trailing-window experiment.
   - The GitHub endpoint also creates survivor bias: stars later removed by
     users are absent from historical counts.

10. **MAPE was unsuitable as a headline metric.**
    - It excludes zero targets and changes the estimand on sparse count series.
    - Final materials de-emphasize it in favor of CRPS, MAE, and supported
      interval coverage.

### Minor corrections and confirmed sound components

- Corrected breakout/7d LightGBM MAE from 15.57 to 15.56.
- Verified all eight summary statistics in the primary CSV against the raw
  parquet across all 54 model/horizon/cohort rows.
- Verified primary CRPS/selected MAE tables and the local-mean sensitivity table.
- Verified cutoff-date consistency, model/window alignment, and equal horizon
  counts.
- Replaced figure titles that implied forecast-time regimes.
- Marked the trailing-window spot-check as separate from the primary backtest.
- The empirical CRPS implementation is the fair finite-sample energy estimator.
- The controlled Normal experiment finds a −0.42% representation shift between
  deterministic quantile grids and random samples; useful reassurance, but not
  a complete domain-matched validation for clipped count forecasts.

## Files and artifacts changed

### Production/evaluation code

- `pulsegraph/forecast/baselines.py`
  - deterministic RNG;
  - last-value, weekly-naive, and local-mean baselines.
- `pulsegraph/forecast/timesfm_forecaster.py`
  - unsupported 95% interval represented as NaN with metadata.
- `pulsegraph/evaluation/metrics.py`
  - finite-bound coverage handling;
  - randomized empirical PIT.
- `scripts/run_experiment.py`
  - preserves the original four baseline defaults;
  - optional `--include-audit-baselines`.

### Review scripts

- `scripts/review_integrity.py`
- `scripts/review_windows.py`
- `scripts/review_scale.py`
- `scripts/review_strong_baselines.py`
- `scripts/review_local_mean_lookbacks.py`
- `scripts/review_cluster_inference.py`
- `scripts/review_censoring.py`

### Final materials

- Reframed and corrected `RESULTS.md` and `WRITEUP.md`.
- Added `reports/pretraining_overlap_audit.md`.
- Added machine-readable scale, clustered-inference, censoring, and
  strong-baseline outputs under `reports/`.
- Replaced the invalid PIT plot with
  `reports/figures/coverage80_by_regime.png`.
- Added `reports/figures/crps_sensitivity_by_cohort.png`.
- Corrected titles/captions in all retained figures.
- Added metric caveats to `reports/experiment_real_full6_meta.json`.

The full foundation-model experiment was **not** rerun. No discovered code bug
required it. New inexpensive baselines were evaluated separately on
verified-identical windows, preserving the original artifact.

## Materials-review confirmation

- `RESULTS.md` numbers match recomputation from raw artifacts.
- `WRITEUP.md` is consistent with the corrected results and no longer claims
  shock-onset prediction, forecast-time regimes, TimesFM 95% calibration, or
  ordinary PIT validity.
- All four retained figures were visually inspected:
  - primary CRPS figure accurately displays the stored six-model means;
  - sensitivity figure displays local mean vs both foundation models;
  - coverage figure displays supported 80% coverage and nominal reference;
  - spot-check figure is explicitly labeled as a trailing-window sanity check,
    not primary evidence.

## Impact assessment

### What a hostile reviewer will attack first

1. **Outcome-conditioned cohorts and early-window placement.** This remains the
   largest scientific weakness and cannot be fixed by prose.
2. **No clean post-training temporal holdout.** No direct contamination was
   found, but a prospective control is stronger than a corpus audit.
3. **Post-hoc baseline construction.** Reviewers can reasonably ask whether
   local-mean choices were adapted after seeing results.
4. **Single platform, selected high-star repositories.** Search ranking and
   star thresholds limit representativeness.
5. **Pseudo-sample CRPS rather than native quantile scoring.** A future run
   should report weighted quantile loss / interval score on common supported
   quantiles alongside CRPS.
6. **Weak probabilistic classical models.** A properly tuned AutoETS and
   count-aware state-space/negative-binomial baseline with rolling calibration
   are still missing.

### Publication readiness

- **Ready now:** technically honest blog post, reproducible empirical note, or
  negative/qualified-results workshop discussion—provided the revised claim is
  retained.
- **Not yet bulletproof for a forecasting venue:** the forecast-time regime and
  temporal-holdout problems require new data/window design, not additional
  analysis of the existing artifact.

### Minimum extension for a defensible workshop paper

1. Freeze model revisions, code, repository manifest, and metrics now.
2. Preregister baseline families and all local-mean lookbacks before evaluating
   new outcomes.
3. Use context-only regime features at each cutoff, or replace the regime claim
   with a formally defined post-launch-decay task.
4. Evaluate trailing, non-overlapping windows and report repository-clustered
   confidence intervals.
5. Add AutoETS and a count-aware probabilistic baseline calibrated on rolling
   residuals.
6. Report common-quantile loss, WIS, MASE/RMSSE, and per-repository skill—not
   MAPE as a primary metric.

### Highest-impact continuation

The most valuable contribution may be a **prospective benchmark for
shock-driven attention dynamics**, not the current model ranking alone:

- publish an immutable repository manifest and aggregate daily-count snapshot;
- define context-only cohort/regime rules;
- include a prospective set of repositories created after model release;
- preserve real zeros through a common observation endpoint;
- provide fixed splits, common quantiles, strong baseline implementations,
  repository-clustered inference, and a lightweight leaderboard;
- extend only after protocol freeze to package downloads, forks, issues, or web
  mentions.

That benchmark would fill a real gap: common forecasting benchmarks are often
smooth and seasonal, while bursty, survivor-biased, zero-inflated public
attention data exposes model robustness and calibration failures. The current
study is useful pilot evidence and infrastructure for that benchmark, but
should not be presented as its final form.

## Prioritized recommendations

1. **P0 — Freeze and preregister a prospective v2 protocol.**
2. **P0 — Redesign window placement and context-only cohort labels.**
3. **P0 — Add strong preregistered classical/count baselines and native
   quantile metrics.**
4. **P1 — Collect the post-release prospective cohort; do not tune on it.**
5. **P1 — Package the aggregate dataset and protocol as a benchmark.**
6. **P1 — Extend to one independent attention domain for external validity.**
7. **P2 — Add event covariates only after the univariate benchmark is frozen.**
8. **P2 — Keep Kronos as a separate out-of-domain transfer study, not part of
   the main PulseGraph result.**
