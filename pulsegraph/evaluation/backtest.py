"""Rolling backtest framework for walk-forward evaluation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Protocol

import numpy as np
import pandas as pd

from pulsegraph.evaluation.metrics import evaluate_forecast
from pulsegraph.forecast.chronos_forecaster import ForecastOutput

logger = logging.getLogger(__name__)


class Forecaster(Protocol):
    """Protocol for any forecaster that produces ForecastOutput."""

    def forecast(
        self,
        series: pd.Series | np.ndarray,
        horizon: int,
        repo_name: str,
    ) -> ForecastOutput: ...


@dataclass
class BacktestConfig:
    """Configuration for rolling backtest."""

    horizons: list[int]
    step_days: int = 7
    min_context_days: int = 90
    max_windows: int | None = None


def rolling_backtest(
    series: pd.Series,
    forecaster: Forecaster,
    config: BacktestConfig,
    repo_name: str = "",
) -> pd.DataFrame:
    """Run a walk-forward backtest on a single series.

    Slides a window forward by `step_days`, using all data up to the cutoff
    as context and evaluating on the next `horizon` days.
    """
    results = []
    values = series.values.astype(np.float64)
    n = len(values)

    max_horizon = max(config.horizons)

    cutoff = config.min_context_days
    window_count = 0

    while cutoff + max_horizon <= n:
        if config.max_windows and window_count >= config.max_windows:
            break

        context = values[:cutoff]

        for horizon in config.horizons:
            if cutoff + horizon > n:
                continue

            actuals = values[cutoff : cutoff + horizon]

            try:
                forecast = forecaster.forecast(context, horizon=horizon, repo_name=repo_name)
                metrics = evaluate_forecast(forecast, actuals)
                metrics["cutoff_idx"] = cutoff
                if hasattr(series.index, "dtype") and pd.api.types.is_datetime64_any_dtype(
                    series.index
                ):
                    metrics["cutoff_date"] = str(series.index[cutoff])
                results.append(metrics)
            except Exception as e:
                logger.warning(
                    "Backtest failed at cutoff=%d, horizon=%d for %s: %s",
                    cutoff, horizon, repo_name, e,
                )

        cutoff += config.step_days
        window_count += 1

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    return df


def run_backtest_suite(
    dataset: dict[str, pd.Series],
    forecasters: dict[str, Forecaster],
    config: BacktestConfig,
    strata: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Run backtests for all repos across all forecasters.

    Args:
        dataset: repo_name -> daily signal Series
        forecasters: model_name -> Forecaster instance
        config: backtest configuration
        strata: optional repo_name -> stratum label mapping
    """
    all_results = []
    total = len(dataset) * len(forecasters)
    done = 0

    for repo_name, series in dataset.items():
        for model_name, forecaster in forecasters.items():
            done += 1
            logger.info("Backtest %d/%d: %s with %s", done, total, repo_name, model_name)

            df = rolling_backtest(series, forecaster, config, repo_name=repo_name)
            if df.empty:
                continue

            df["model_name"] = model_name
            if strata and repo_name in strata:
                df["stratum"] = strata[repo_name]
            all_results.append(df)

    if not all_results:
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)
    return combined


def summarize_backtest(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate backtest results by model, horizon, and stratum."""
    if results_df.empty:
        return pd.DataFrame()

    group_cols = ["model_name", "horizon"]
    if "stratum" in results_df.columns:
        group_cols.append("stratum")

    summary = results_df.groupby(group_cols).agg(
        crps_mean=("crps_mean", "mean"),
        crps_std=("crps_mean", "std"),
        mape_mean=("mape", "mean"),
        mape_median=("mape", "median"),
        mae_mean=("mae", "mean"),
        coverage_80_mean=("coverage_80", "mean"),
        coverage_95_mean=("coverage_95", "mean"),
        n_windows=("crps_mean", "count"),
    ).reset_index()

    return summary
