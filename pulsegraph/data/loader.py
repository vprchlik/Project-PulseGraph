"""Data loading utilities for working with local parquet/CSV files.

Provides a unified interface for loading time-series data regardless of
whether it came from BigQuery, the GitHub API, or local cache.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pulsegraph.config import RAW_DIR, PROCESSED_DIR

logger = logging.getLogger(__name__)


def load_daily_signals(path: Path | None = None) -> pd.DataFrame:
    """Load the daily signals parquet file."""
    if path is None:
        path = RAW_DIR / "daily_signals.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Daily signals file not found at {path}. "
            "Run the GH Archive backfill first (scripts/backfill_gharchive.py)."
        )
    df = pd.read_parquet(path)
    df["event_date"] = pd.to_datetime(df["event_date"])
    return df


def load_target_repos(path: Path | None = None) -> pd.DataFrame:
    """Load the target repos list."""
    if path is None:
        path = RAW_DIR / "target_repos.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Target repos file not found at {path}.")
    return pd.read_parquet(path)


def prepare_star_series(
    signals_df: pd.DataFrame,
    repo_name: str,
    signal_col: str = "stars",
    min_history_days: int = 90,
) -> pd.Series | None:
    """Extract a single repo's daily signal as a contiguous time series.

    Fills missing dates with 0 and returns a DatetimeIndex-ed Series.
    Returns None if the repo has insufficient history.
    """
    repo_data = signals_df[signals_df["repo_name"] == repo_name].copy()
    if repo_data.empty:
        return None

    repo_data = repo_data.sort_values("event_date")
    repo_data = repo_data.set_index("event_date")

    if signal_col not in repo_data.columns:
        return None

    date_range = pd.date_range(
        start=repo_data.index.min(),
        end=repo_data.index.max(),
        freq="D",
    )
    series = repo_data[signal_col].reindex(date_range, fill_value=0)
    series.index.name = "date"

    if len(series) < min_history_days:
        return None

    return series


def prepare_cumulative_stars(
    signals_df: pd.DataFrame,
    repo_name: str,
) -> pd.Series | None:
    """Build a cumulative star count series from daily deltas."""
    daily = prepare_star_series(signals_df, repo_name, "stars")
    if daily is None:
        return None
    return daily.cumsum()


def build_dataset_for_forecasting(
    signals_df: pd.DataFrame,
    repo_names: list[str],
    signal_col: str = "stars",
    min_history_days: int = 90,
) -> dict[str, pd.Series]:
    """Build a dict of repo_name -> daily signal Series for forecasting."""
    dataset = {}
    for name in repo_names:
        series = prepare_star_series(signals_df, name, signal_col, min_history_days)
        if series is not None:
            dataset[name] = series
    logger.info(
        "Built forecasting dataset: %d/%d repos had sufficient history",
        len(dataset), len(repo_names),
    )
    return dataset


def generate_synthetic_data(
    n_repos: int = 100,
    n_days: int = 365 * 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic daily star data for testing without BigQuery access.

    Produces realistic-ish star count patterns: base rate + trend + noise +
    occasional spikes simulating viral events.
    """
    rng = np.random.default_rng(seed)
    records = []

    start_date = pd.Timestamp("2022-01-01")
    dates = pd.date_range(start_date, periods=n_days, freq="D")

    for i in range(n_repos):
        base_rate = rng.exponential(5)
        trend = rng.uniform(-0.01, 0.05)
        noise_scale = max(1, base_rate * 0.3)

        daily = np.maximum(
            0,
            base_rate
            + trend * np.arange(n_days)
            + rng.normal(0, noise_scale, n_days),
        ).astype(int)

        n_spikes = rng.poisson(3)
        for _ in range(n_spikes):
            spike_day = rng.integers(0, n_days)
            spike_mag = rng.exponential(base_rate * 10)
            spike_len = rng.integers(1, 14)
            end = min(spike_day + spike_len, n_days)
            decay = np.exp(-np.arange(end - spike_day) * 0.3)
            daily[spike_day:end] += (spike_mag * decay).astype(int)

        for d, val in zip(dates, daily):
            records.append({
                "repo_name": f"synthetic-org/repo-{i:04d}",
                "event_date": d,
                "stars": int(val),
                "forks": max(0, int(val * rng.uniform(0.05, 0.2))),
                "pushes": max(0, int(rng.poisson(base_rate * 0.5))),
                "issues": max(0, int(rng.poisson(base_rate * 0.1))),
                "pull_requests": max(0, int(rng.poisson(base_rate * 0.05))),
                "releases": 1 if rng.random() < 0.01 else 0,
            })

    df = pd.DataFrame(records)
    logger.info("Generated synthetic data: %d repos, %d days, %d rows", n_repos, n_days, len(df))
    return df
