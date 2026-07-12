#!/usr/bin/env python3
"""Sensitivity of the local-mean baseline to its lookback choice.

The 28-day baseline was added after the primary result was known. To avoid
presenting one convenient lookback as uniquely favorable, this script evaluates
five ordinary calendar-scale choices on identical windows.
"""

from __future__ import annotations

import pandas as pd

from pulsegraph.config import RAW_DIR, REPORTS_DIR
from pulsegraph.data.loader import build_dataset_for_forecasting, load_daily_signals
from pulsegraph.evaluation.backtest import BacktestConfig, run_backtest_suite
from pulsegraph.forecast.baselines import LocalMeanForecaster

LOOKBACKS = [7, 14, 28, 56, 90]
CONFIG = BacktestConfig(
    horizons=[7, 30, 90],
    step_days=30,
    min_context_days=180,
    max_windows=8,
)


class _NamedLocalMean(LocalMeanForecaster):
    def forecast(self, *args, **kwargs):
        output = super().forecast(*args, **kwargs)
        output.model_name = f"local_mean_{self.lookback}"
        return output


def main() -> None:
    metadata = pd.read_parquet(RAW_DIR / "ingest_metadata.parquet")
    metadata = metadata[metadata["status"] == "ok"]
    strata = dict(zip(metadata["repo_name"], metadata["regime_label"]))
    dataset = build_dataset_for_forecasting(
        load_daily_signals(RAW_DIR / "daily_signals.parquet"),
        list(strata),
        signal_col="stars",
        min_history_days=270,
    )
    models = {
        f"local_mean_{lookback}": _NamedLocalMean(lookback=lookback, n_samples=200)
        for lookback in LOOKBACKS
    }
    results = run_backtest_suite(dataset, models, CONFIG, strata=strata)
    path = REPORTS_DIR / "review_local_mean_lookbacks.parquet"
    results.to_parquet(path, index=False)
    table = (
        results.groupby(["stratum", "horizon", "model_name"])["crps_mean"]
        .mean()
        .unstack("model_name")
    )
    print(table.round(3).to_string())
    print("\nBest lookback in each cohort/horizon cell:")
    print(table.idxmin(axis=1).to_string())
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
