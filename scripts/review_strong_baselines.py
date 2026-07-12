#!/usr/bin/env python3
"""Post-hoc sensitivity check with stronger conventional baselines.

The preregistered-style main run used four draft baselines, but its
``naive_seasonal`` changes period with forecast horizon. This audit adds three
fixed, deterministic alternatives on the exact same repositories and cutoffs:

* last value / random walk;
* fixed weekly seasonal naive; and
* a recent 28-day local mean.

The output is deliberately stored as a separate review artifact rather than
silently rewriting the primary six-model result.
"""

from __future__ import annotations

import pandas as pd

from pulsegraph.config import RAW_DIR, REPORTS_DIR
from pulsegraph.data.loader import build_dataset_for_forecasting, load_daily_signals
from pulsegraph.evaluation.backtest import BacktestConfig, run_backtest_suite
from pulsegraph.forecast.baselines import (
    LastValueForecaster,
    LocalMeanForecaster,
    WeeklySeasonalNaiveForecaster,
)

N_SAMPLES = 200
CONFIG = BacktestConfig(
    horizons=[7, 30, 90],
    step_days=30,
    min_context_days=180,
    max_windows=8,
)


def main() -> None:
    metadata = pd.read_parquet(RAW_DIR / "ingest_metadata.parquet")
    metadata = metadata[metadata["status"] == "ok"]
    strata = dict(zip(metadata["repo_name"], metadata["regime_label"]))
    signals = load_daily_signals(RAW_DIR / "daily_signals.parquet")
    dataset = build_dataset_for_forecasting(
        signals,
        list(strata),
        signal_col="stars",
        min_history_days=270,
    )
    models = {
        "last_value": LastValueForecaster(n_samples=N_SAMPLES),
        "weekly_naive": WeeklySeasonalNaiveForecaster(n_samples=N_SAMPLES),
        "local_mean": LocalMeanForecaster(n_samples=N_SAMPLES),
    }
    audit = run_backtest_suite(dataset, models, CONFIG, strata=strata)
    out = REPORTS_DIR / "review_strong_baselines_results.parquet"
    audit.to_parquet(out, index=False)

    primary = pd.read_parquet(REPORTS_DIR / "experiment_real_full6_results.parquet")
    key = ["repo_name", "horizon", "cutoff_idx", "stratum"]
    primary_keys = set(map(tuple, primary[key].drop_duplicates().to_numpy()))
    audit_keys = set(map(tuple, audit[key].drop_duplicates().to_numpy()))
    if primary_keys != audit_keys:
        raise RuntimeError("Audit baseline windows do not match primary result windows")

    combined = pd.concat([primary, audit], ignore_index=True)
    summary = (
        combined.groupby(["model_name", "horizon", "stratum"])
        .agg(
            crps_mean=("crps_mean", "mean"),
            mae_mean=("mae", "mean"),
            coverage_80_mean=("coverage_80", "mean"),
            n_windows=("crps_mean", "count"),
        )
        .reset_index()
    )
    summary.to_csv(REPORTS_DIR / "review_nine_model_summary.csv", index=False)

    models_order = [
        "naive_seasonal",
        "last_value",
        "weekly_naive",
        "local_mean",
        "linear",
        "ets",
        "lightgbm",
        "chronos2",
        "timesfm",
    ]
    table = (
        combined.groupby(["stratum", "horizon", "model_name"])["crps_mean"]
        .mean()
        .unstack("model_name")[models_order]
    )
    print("Mean CRPS sensitivity table (primary + post-hoc audit baselines):")
    print(table.round(3).to_string())

    repo = (
        combined.groupby(["stratum", "horizon", "repo_name", "model_name"])["crps_mean"]
        .mean()
        .unstack("model_name")
    )
    baselines = models_order[:7]
    print("\nFoundation model repository win-rate vs best *fixed model* baseline:")
    for (stratum, horizon), group in repo.groupby(level=["stratum", "horizon"]):
        fixed_best_name = str(group[baselines].mean().idxmin())
        for model in ["chronos2", "timesfm"]:
            rate = float((group[model] < group[fixed_best_name]).mean())
            print(
                f"{stratum:9s} h={int(horizon):2d}d {model:8s} "
                f"vs {fixed_best_name:15s}: {rate:5.1%}"
            )

    print(f"\nSaved {out}")
    print(f"Saved {REPORTS_DIR / 'review_nine_model_summary.csv'}")


if __name__ == "__main__":
    main()
