# Do Time-Series Foundation Models Help for Shock-Driven Attention Signals?

*A small, honest empirical study on forecasting daily GitHub stars.*

> Draft technical report — structured for later development into a blog post or
> workshop submission. Numbers and figures are sourced from `RESULTS.md` and
> `reports/`.

---

## Abstract

We test whether zero-shot time-series foundation models (Chronos-2 and
TimesFM-2.5) beat conventional methods at forecasting **daily new GitHub
stars**. On a primary rolling-origin backtest over 179 repositories, both models
have lower mean CRPS than four initial baselines in every discovery
cohort/horizon cell. An adversarial audit substantially qualifies that result.
A post-hoc 28-day local mean closes much of the gap; repository-clustered
inference finds reliable gains in the breakout-selected cohort and parts of the
dead-selected cohort, but no typical-repository advantage in the
steady-selected cohort. The gain is mainly robustness against a minority of
large errors. Because 78% of breakout-selected repositories peak before
evaluation, this is evidence about post-breakout dynamics—not evidence that
foundation models predict viral shocks before they occur.

---

## 1. Motivation

Attention signals for public entities—GitHub stars, package downloads, and
mentions—often contain launch effects, bursts, and long decays. Time-series
foundation models may provide useful zero-shot priors for these shapes. The
practical question is:

> Are foundation models worth the compute here, and if so, when?

The audit revealed that the current experiment does not place most test windows
at shock onset. We therefore treat "shock-driven" as a property of the source
domain, not as a claim that this design evaluates shock prediction.

## 2. Method

**Data.** Real daily new-star counts reconstructed from GitHub REST stargazer
timestamps. Of 208 repositories ingested, 179 have at least 270 days of
history. Reproducible Search-API queries form three **discovery-time,
outcome-conditioned cohorts**: breakout-selected (41 usable), steady-selected
(69), and dead-selected (69). These labels are not forecast-time states.

**Task.** One-shot multi-horizon probabilistic forecasting of the daily-stars
series at h ∈ {7, 30, 90} days.

**Backtest.** Walk-forward rolling origin beginning 180 days after each
repository's first observed star, step 30 days, and at most eight windows.
Models receive only past target values. This direct indexing is leakage-free,
but the cohort labels themselves use later outcomes.

**Models.** The primary run contains horizon-dependent naive seasonal, linear,
damped ETS, LightGBM, Chronos-2, and TimesFM-2.5. A post-hoc audit adds
deterministic last-value, fixed weekly seasonal-naive, and 28-day local-mean
baselines on identical windows. Quantile outputs from the foundation models are
converted to 200 pseudo-samples by inverse-CDF interpolation.

**Metrics.** Empirical CRPS, MAE, and supported 80% interval coverage. TimesFM
exposes q0.1–q0.9 only, so no TimesFM 95% interval is reported. Archived
non-randomized PIT values are not interpreted because ordinary PIT is invalid
for tied discrete counts; future runs use randomized empirical PIT.

**Validation.** The audit verifies common window keys and summary
recomputation; checks scale-normalized and equal-repository scores; adds stronger
baselines; averages overlapping windows within repository before bootstrap and
Wilcoxon inference; applies Benjamini-Hochberg correction; quantifies window
placement and endpoint censoring; and reviews disclosed pretraining corpora.

## 3. Results

**Primary result.** Both foundation models have lower mean CRPS than the four
original baselines in every cohort/horizon cell.

**Stronger-baseline sensitivity.** A 28-day local mean is substantially stronger
than every original baseline. The foundation models still retain lower mean
CRPS in all cells, but repository win rates fall to 40.6–80.5%. After
repository-clustered inference and multiplicity correction:

- TimesFM is better than local mean across the three breakout-selected horizons;
  Chronos is better at 7/30d but not 90d.
- TimesFM is better across the three dead-selected horizons; Chronos at 7/30d
  but not 90d.
- Neither foundation model shows a reliable typical-repository advantage in the
  steady-selected cohort.

The lower foundation-model cell means persist across local-mean lookbacks of
7, 14, 28, 56, and 90 days; the post-hoc baseline conclusion is not unique to
the chosen 28-day window.

The bootstrap mean difference remains favorable more broadly because a minority
of repositories contribute large baseline errors. The most defensible mechanism
is **robustness against large misses**, not uniform superiority.

**Scale robustness.** Equal-repository macro means preserve the primary ranking,
and dropping any one repository never changes a foundation-model winner into a
baseline winner. The improvement is not solely a scale artifact, although the
largest five repositories account for a substantial share of positive gains in
some cells.

**Calibration.** At the mutually supported 80% level, TimesFM coverage is
0.774–0.852 and Chronos coverage is 0.729–0.784. Chronos supports a wider 95%
interval, but TimesFM does not expose the required tail quantiles, so a direct
95% comparison is unavailable.

**Cohort interpretation.** For 78% of breakout-selected repositories, the
global peak is before evaluation. Dead-selected windows have a median 5.02
stars/day and are not generally flat. The cohort analysis therefore concerns
early-life/post-peak trajectories conditioned on later outcomes.

Full tables and figures: `RESULTS.md §4`; `reports/figures/`.

## 4. Limitations

- **Outcome-conditioned cohorts.** Discovery labels use later maintenance,
  inactivity, or success and are unavailable at forecast time.
- **Early-window placement.** The design mostly evaluates post-peak decay and
  early lifecycle, not shock onset or current dead/steady states.
- **Post-hoc baseline strengthening.** Local mean was added after observing the
  primary result. It is a transparent sensitivity check, not preregistered.
- **Pretraining overlap cannot be excluded completely.** Disclosed corpora do not
  list GitHub stars, but full item-level cutoffs are not independently
  auditable. None of the currently eligible repositories was created after both
  model releases.
- **Archived reproducibility.** Primary baseline residual draws were unseeded.
  Future code is deterministic, but the stored primary artifact is not
  bit-for-bit reproducible.
- **Endpoint censoring.** Series run from first to last observed star rather than
  repository creation to snapshot. Trailing gaps are generally small but must
  be handled explicitly in a future trailing-window design.
- **Current-stargazer survivor bias.** Stars later removed by users are absent
  from the API snapshot, so reconstructed historical counts omit un-starred
  events.
- **Single signal and platform; zero-shot only.** External validity and the
  effect of domain fine-tuning remain unknown.
- **No event covariates.** Releases and external mentions are likely required to
  predict shock occurrence rather than decay.

## 5. Future work

1. **Preregister a prospective holdout.** Freeze model weights and protocol,
   then evaluate repositories created after model release. This is the cleanest
   contamination and selection-bias control.
2. **Redesign regime analysis.** Use trailing windows and context-only regime
   features at each cutoff. Do not label an early window with a later outcome.
3. **Strengthen baselines before new foundation-model runs.** Include local
   level/mean, fixed weekly naive, AutoETS, and a count-aware probabilistic
   baseline with rolling/out-of-fold residual calibration.
4. **Publish a benchmark protocol.** Release immutable repository lists,
   timestamped snapshots, cohort-selection rules, normalized skill metrics, and
   repository-clustered uncertainty.
5. Add event covariates and other attention signals only after the core
   prospective protocol is fixed.

---

*Companion artifacts: `RESULTS.md` (full results), `reports/` (raw parquet,
summaries, calibration, figures), `scripts/` (reproducible pipeline).*
