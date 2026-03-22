"""Shared application state for the API server.

Holds loaded models, cached data, and the analog index so they persist
across requests without re-initialization.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pulsegraph.config import RAW_DIR, settings
from pulsegraph.data.loader import build_dataset_for_forecasting, load_daily_signals
from pulsegraph.regime.analog import AnalogIndex

logger = logging.getLogger(__name__)


class AppState:
    """Holds all shared state for the running API."""

    def __init__(self):
        self.signals_df: pd.DataFrame | None = None
        self.dataset: dict[str, pd.Series] = {}
        self.repo_metadata: dict[str, dict] = {}
        self.analog_index: AnalogIndex = AnalogIndex()
        self.forecaster = None
        self.model_loaded = False
        self._forecast_cache: dict[str, dict] = {}

    def load(self):
        """Load data and models on startup."""
        signals_path = RAW_DIR / "daily_signals.parquet"
        if signals_path.exists():
            self.signals_df = load_daily_signals(signals_path)
            repo_names = self.signals_df["repo_name"].unique().tolist()
            self.dataset = build_dataset_for_forecasting(
                self.signals_df, repo_names, min_history_days=30
            )
            logger.info("Loaded %d repos with sufficient history", len(self.dataset))

            self._build_analog_index()
        else:
            logger.warning("No signals data found at %s", signals_path)

        repos_path = RAW_DIR / "target_repos.parquet"
        if repos_path.exists():
            repos_df = pd.read_parquet(repos_path)
            for _, row in repos_df.iterrows():
                name = row.get("repo_name", "")
                self.repo_metadata[name] = row.to_dict()

        # Try to import Chronos; model weights are loaded lazily on first forecast call.
        # This keeps startup fast and avoids blocking on HuggingFace at boot time.
        try:
            from pulsegraph.forecast.chronos_forecaster import ChronosForecaster  # noqa: F401

            self.forecaster = ChronosForecaster()
            self.model_loaded = True
            logger.info("Chronos-2 forecaster initialized (weights loaded on first use)")
        except Exception as e:
            logger.warning("Chronos-2 not available: %s. Using ETS fallback.", e)
            from pulsegraph.forecast.baselines import ETSForecaster

            self.forecaster = ETSForecaster(n_samples=settings.trajectory_samples)
            self.model_loaded = False

    def _build_analog_index(self):
        """Build the analog index from all available series."""
        logger.info("Building analog index from %d series", len(self.dataset))
        for name, series in self.dataset.items():
            self.analog_index.add_states_from_series(name, series, window_size=30, step=14)
        logger.info("Analog index: %d states", len(self.analog_index.states))

    def get_forecast(self, repo_name: str, horizon: int) -> dict | None:
        """Get or compute a forecast for a repo."""
        cache_key = f"{repo_name}_{horizon}"
        if cache_key in self._forecast_cache:
            return self._forecast_cache[cache_key]

        if repo_name not in self.dataset:
            return None

        series = self.dataset[repo_name]

        from pulsegraph.regime.sampler import analyze_regimes

        try:
            forecast = self.forecaster.forecast(series, horizon=horizon, repo_name=repo_name)
        except Exception as e:
            # Chronos weights unavailable (no internet / no GPU): fall back to ETS
            logger.warning("Chronos forecast failed (%s); falling back to ETS", e)
            from pulsegraph.forecast.baselines import ETSForecaster

            fallback = ETSForecaster(n_samples=settings.trajectory_samples)
            self.forecaster = fallback
            self.model_loaded = False
            forecast = fallback.forecast(series, horizon=horizon, repo_name=repo_name)

        regime = analyze_regimes(forecast)

        result = {
            "forecast": forecast,
            "regime": regime,
        }

        self._forecast_cache[cache_key] = result
        return result

    def clear_cache(self):
        self._forecast_cache.clear()
