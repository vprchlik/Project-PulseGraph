# Audit handoff (stopped at user request)

> Superseded by the completed audit in `STUDY_AUDIT.md`. This file is retained
> as an exact record of the interrupted audit state.

Date stopped: 2026-07-12. This is a partial audit, not a final scientific verdict.
The user explicitly stopped further investigation to control cost.

## 1. Work completed

### Files inspected

- `pulsegraph/data/loader.py`
- `pulsegraph/forecast/baselines.py`
- `pulsegraph/forecast/chronos_forecaster.py`
- `pulsegraph/forecast/timesfm_forecaster.py`
- `pulsegraph/evaluation/backtest.py`
- `pulsegraph/evaluation/metrics.py`
- `pulsegraph/evaluation/calibration.py`
- `scripts/run_experiment.py`
- `scripts/add_timesfm.py`
- `scripts/ingest_stars.py`
- `scripts/validate_chronos.py`
- `scripts/test_crps_fairness.py`
- `scripts/final_analysis.py`
- `RESULTS.md`
- `WRITEUP.md`
- `reports/experiment_real_full6_meta.json`
- `reports/experiment_real_full6_summary.csv`

The raw results parquet and raw signal/metadata parquets were read by the two
review scripts described below.

### Commands/checks run

1. Repository/artifact inventory:

   ```bash
   ls -la && ls pulsegraph/ scripts/ reports/ reports/figures/ tests/
   ```

2. Run-integrity audit:

   ```bash
   .venv/bin/python scripts/review_integrity.py
   ```

   Results:

   - 24,732 rows; six expected models.
   - All six models have exactly the same 4,122 `(repo, horizon, cutoff_idx)`
     keys (1,374 windows per horizon).
   - No duplicate model/window rows.
   - Every repo has one consistent stratum.
   - Recomputed summary values matched the CSV for six checked aggregation
     columns (`crps_mean`, `mape_mean`, `mae_mean`, both coverage means, and
     `n_windows`) across all 54 model/horizon/stratum rows. `crps_std` and
     `mape_median` were not yet independently checked.
   - The CRPS table in `RESULTS.md` matched recomputation.
   - The selected MAE table matched except one rounding discrepancy:
     breakout/7d LightGBM is 15.564631, which rounds to 15.56 under the check,
     while the document says 15.57.
   - Coverage ranges quoted in `RESULTS.md` matched the parquet.
   - `cutoff_date` was consistent within every `(repo, cutoff_idx)`.

3. Window-placement audit:

   ```bash
   .venv/bin/python scripts/review_windows.py
   ```

   Results:

   - Breakout evaluation spans roughly 2024-07 to 2026-06; median start
     2025-07-09 and median end 2026-05-04.
   - Steady evaluation spans roughly 2009-05 to 2021-03; median start
     2015-04-10 and median end 2016-02-03.
   - Dead evaluation spans roughly 2009-06 to 2022-04; median start
     2015-10-04 and median end 2016-07-29.
   - Global series peak was before the evaluated span for 78.0% of breakout
     repos; only 12.2% had the global peak inside the evaluated span and 9.8%
     after it.
   - For dead repos, 68.1% had the global peak before evaluation, 7.2% inside,
     and 24.6% after.
   - Median evaluated zero-day fractions were breakout 0.3%, steady 34.7%,
     dead 7.7%.
   - Median evaluated means were breakout 14.83, steady 1.90, dead 5.02
     stars/day. This directly contradicts describing the evaluated dead windows
     as nearly zero/flat.
   - Of breakout window cutoffs, 100% were in/after 2024 and 66.7% were on/after
     2025-06. Steady/dead cutoffs were all before 2024.

4. Initial official-source web searches (not yet developed into a complete
   contamination audit):

   - Chronos-2: Amazon release dated 2025-10-20; model card says training used a
     subset of Chronos datasets, a subset of GIFT-Eval Pretrain, and synthetic
     data. Exact real-data cutoffs and whether any GitHub attention series were
     included remain unresolved.
   - TimesFM-2.5: Google release dated 2025-09-15; model card lists
     GIFT-EvalPretrain, Wikimedia Pageviews (cutoff Nov 2023), Google Trends
     top queries (cutoff end of 2022), and synthetic/augmented data. Direct
     GitHub-star inclusion has not been established.
   - Sources found:
     - https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting
     - https://huggingface.co/amazon/chronos-2/blob/main/README.md
     - https://arxiv.org/abs/2510.15821
     - https://github.com/google-research/timesfm
     - https://huggingface.co/google/timesfm-2.5-200m-pytorch

## 2. Preliminary findings, severity-ranked

### Invalidating (for specific claims; no confirmed invalidation of the raw overall comparison)

1. **The regime-specific interpretation is invalid as currently phrased.**
   Backtesting always starts 180 days after each repo's first observed star and
   stops after at most eight 30-day steps. For old steady/dead repos, this tests
   their early lifecycle years before the discovery-time label. The label
   "dead" is assigned using future inactivity, but its evaluated windows have a
   median 5.02 stars/day and only 7.7% zero days. Therefore statements such as
   "dead (flat)", "all models are close because the series is nearly zero", and
   conclusions about forecast-time dead/steady regimes are not supported by the
   evaluated windows. The aggregate comparison on this outcome-selected sample
   may remain numerically valid, but the advertised regime analysis does not.

2. **The experiment mostly tests post-spike decay, not forecasting shocks.**
   For 78% of breakout repos, the global peak is already in context before the
   first evaluated day; only 12.2% place the peak in the whole evaluated span.
   Thus "FMs win on breakout series" currently means mostly "FMs forecast the
   early post-viral decay of repos selected for having already broken out." It
   does not demonstrate predicting shock occurrence or performance through a
   future breakout. This substantially narrows the headline scientific claim.

### Must-disclose / must-fix before submission

1. **Outcome-conditioned cohort selection.** Breakout is selected for already
   having high stars; dead is selected for later inactivity; steady is selected
   for later maintenance. This is not within-window model-input leakage, but it
   is future-conditioned dataset construction and makes regime labels
   unavailable at forecast time. It limits external validity and can shape the
   trajectories being compared.

2. **Pretraining overlap/memorization is unresolved.** Most breakout evaluation
   occurs before or around the 2025 model releases, and some extends into 2026.
   The declared corpora make direct GitHub-star memorization seem unproven, not
   impossible. Chronos-2's real-data cutoffs need authoritative verification;
   GIFT/Chronos corpus contents need checking. This must be a prominent
   limitation unless ruled out. A clean control would evaluate only observations
   after an established training cutoff, preferably repos created after it.

3. **TimesFM does not supply a valid 95% interval in this implementation.**
   Its wrapper only has q0.1...q0.9 and clamps probabilities outside that range.
   Consequently the reported 2.5th/97.5th percentiles collapse to approximately
   q0.1/q0.9: effectively an 80% model interval labeled "95%." The observed
   0.82-0.86 coverage cannot be interpreted as TimesFM 95% calibration or
   ordinary 95% undercoverage. `RESULTS.md`/`WRITEUP.md` currently overinterpret
   this as tail calibration.

4. **PIT/KS calibration is not valid for discrete zero-inflated counts as
   implemented.** `pit = mean(samples <= actual)` is the ordinary PIT with a
   deterministic tie convention. For discrete outcomes, especially with
   nonnegative clipping and many zeros/tied pseudo-samples, PIT is not Uniform
   even under correct calibration. A randomized PIT (or suitable discrete PIT)
   and cluster-aware interpretation are needed. Per-window KS p-values are also
   not useful with only 7/30/90 dependent horizon steps.

5. **Final six-model artifact is stitched, though on identical windows.**
   `scripts/add_timesfm.py` appends a separately run TimesFM result to
   `experiment_real_full_results.parquet`. This is scientifically acceptable
   only because window keys/config match, which the audit confirmed, but it is
   not literally one six-model execution. Baseline residual sampling is
   unseeded, so exact reruns are not reproducible and the old baseline results
   cannot be regenerated bit-for-bit without stored RNG state.

6. **Baseline set is weaker than the wording suggests.**
   `naive_seasonal` repeats the last *horizon* days, so its period changes from
   7 to 30 to 90. It is not a conventional fixed weekly seasonal-naive baseline.
   There is no last-value/random-walk baseline, fixed 7-day seasonal naive,
   count model, or tuned/validated AutoETS. ETS has no weekly seasonality.
   These omissions create a credible strawman-baseline attack.

7. **LightGBM uncertainty attribution needs correction.** Residuals are
   in-sample training residuals from a flexible model, sampled i.i.d. around one
   deterministic recursively generated path. This predictably yields intervals
   that are too narrow; it omits model/recursive uncertainty. Calling the issue
   merely "residual bootstrap on training residuals" is incomplete. Out-of-fold
   or rolling residuals and trajectory-recursive innovations are needed for a
   fair probabilistic baseline.

8. **Overlapping-window inference is not yet rigorous.** Wilcoxon tests treat
   overlapping windows and repeated repos as if paired observations were
   independent. Horizon-90 forecasts overlap heavily at a 30-day step. Venue
   review will demand repo-clustered bootstrap/permutation or Diebold-Mariano
   style tests with HAC/cluster handling, plus multiplicity control.

9. **Scale domination remains untested.** Mean raw CRPS/MAE can be dominated by
   high-volume repos. No per-series normalized score or skill score was run
   before the stop request. This is a priority because it could materially
   weaken the positive mean result.

10. **MAPE is unsuitable for the headline task.** It excludes zero actuals,
    which changes the estimand on zero-inflated series, and averages per-window
    percentages. It should be de-emphasized or replaced by WAPE/MASE/RMSSE with
    clearly defined denominators.

11. **Series endpoint handling needs audit.** Filling missing dates with zero
    between first and last observed stars is correct for complete event data.
    However, `prepare_star_series` starts at first star and ends at last star,
    not repo creation and ingestion dates. Potential leading/trailing zero runs
    are censored. This was not quantified before stopping.

### Minor / confirmed sound so far

1. Backtest indexing itself uses only `values[:cutoff]` and evaluates
   `values[cutoff:cutoff+horizon]`; no direct look-ahead was found.
2. All models use identical window keys; no missing model/window rows were found.
3. The summary CSV matched the raw parquet for the six aggregation fields
   checked.
4. The main CRPS table and quoted coverage ranges match the raw results.
5. "TimesFM marginally better central accuracy" is numerically supported by the
   summary: TimesFM has lower CRPS in 8/9 cells and lower mean MAE in 9/9 versus
   Chronos-2. The tail-calibration half of that comparison is not valid for the
   reason above.
6. One document rounding issue: breakout/7d LightGBM MAE should be 15.56 under
   ordinary two-decimal rounding, not 15.57.
7. The controlled Normal CRPS representation test may be reassuring, but it
   does not by itself validate deterministic quantile-grid scoring for clipped,
   discrete, zero-inflated distributions. This needs either a domain-matched
   check or direct quantile-score CRPS approximation.

## 3. Files changed and artifacts created

No production code or final documentation was edited.

Created:

- `scripts/review_integrity.py`
  - Read-only audit script for model/window alignment, duplicates, stratum
    consistency, partial summary recomputation, selected document table values,
    coverage ranges, and cutoff-date consistency.
  - Verified by running successfully with exit code 0.

- `scripts/review_windows.py`
  - Read-only audit script reconstructing the forecasting dataset and measuring
    calendar placement, global-peak placement, zero fractions, and scale in the
    evaluated spans.
  - Verified by running successfully with exit code 0.

- `AUDIT_HANDOFF.md`
  - This handoff.

No files were reverted or deleted. Preserve the two `review_*.py` scripts.

## 4. Work still outstanding (prioritized)

- [ ] **P0:** Reframe or redesign regime evaluation. At minimum, document that
      labels are discovery-time outcomes while windows are early-life. Prefer
      evaluating recent/trailing rolling windows and/or assigning regime using
      context-only features at each cutoff.
- [ ] **P0:** Compute scale-robust results: per-repo normalized CRPS, MASE/RMSSE,
      and skill versus fixed weekly seasonal-naive and last-value baselines.
      Report macro-averages by repo as well as window-weighted means.
- [ ] **P0:** Correct TimesFM 95% calibration claims; do not label q10-q90 as a
      95% interval. Check whether the continuous quantile API can directly
      request 0.025/0.975; otherwise report only supported central coverage.
- [ ] **P0:** Replace ordinary PIT with randomized/discrete PIT or remove strong
      PIT-uniformity claims.
- [ ] **P0:** Complete authoritative pretraining-data/cutoff audit for both
      models and add a prominent contamination limitation/control.
- [ ] **P1:** Add competent fixed baselines (last value, weekly seasonal naive,
      robust local level/trend, properly seasonal AutoETS, and ideally a
      count-aware model) without tuning on test outcomes.
- [ ] **P1:** Rework LightGBM probabilistic forecasts using rolling/out-of-fold
      residuals and recursive trajectory simulation; seed all stochastic paths.
- [ ] **P1:** Recompute statistical inference at repo level or with
      cluster/HAC-aware methods; account for overlapping windows and multiple
      comparisons.
- [ ] **P1:** Quantify scale domination directly (top repos' contribution to
      total CRPS gap; leave-one-repo-out sensitivity; median repo skill).
- [ ] **P1:** Audit leading/trailing censoring relative to repository creation
      and ingestion timestamp; decide the correct observation interval.
- [ ] **P1:** Verify every remaining number in both docs, including oracle
      win-rates/p-values and MAE decomposition. The current integrity script did
      not verify all prose or all summary columns.
- [ ] **P1:** Visually inspect all three figures and verify captions. Only file
      existence was established before stopping.
- [ ] **P1:** Rewrite `RESULTS.md` and `WRITEUP.md` limitations and verdict after
      the above analyses. Remove unsupported "nearly zero dead" and
      "shock forecasting" language.
- [ ] **P2:** Add unit tests for wrapper quantile behavior, deterministic seeds,
      discrete PIT, and window-selection policy.
- [ ] **P2:** Run the complete test suite only after edits and report the count.

## 5. Exact continuation prompt for a future agent

> Resume the adversarial scientific audit of `/home/vprchlik/projects/PulseGraph`
> on `main`, leaving `../PulseGraph-kronos` untouched. First read
> `AUDIT_HANDOFF.md`; do not repeat completed inspections or rerun
> `scripts/review_integrity.py` / `scripts/review_windows.py` unless a relevant
> input changed. Preserve those scripts. Start with the P0 checklist: (1)
> scale-normalized/macro-per-repo CRPS and skill scores versus fixed weekly
> seasonal-naive and last-value; (2) verify whether TimesFM can request true
> 0.025/0.975 quantiles and correct invalid 95% claims; (3) implement or
> appropriately qualify discrete randomized PIT; (4) complete official-source
> Chronos-2/TimesFM training-corpus and cutoff analysis; (5) determine whether
> recent/trailing windows or context-only regime labels materially change the
> regime conclusions. Be skeptical and report negative results. Do not rerun
> the full foundation-model experiment unless a confirmed invalidating code
> bug requires it; prefer analyses from the existing parquet and minimum
> targeted reruns. Then verify every number and figure, edit only in-scope
> forecasting code/docs, and run `.venv/bin/python -m pytest tests/ -q` at the
> end. Treat the current regime-specific headline as unsupported until repaired,
> while keeping the distinction that the raw aggregate six-model comparison has
> not yet been disproved.

## 6. Test status

No test suite was run during this partial audit. The user-provided starting state
was 26/26 tests passing via `.venv/bin/python -m pytest tests/ -q`; that status
was not independently re-verified after creating the two read-only review
scripts. No production code was changed.
