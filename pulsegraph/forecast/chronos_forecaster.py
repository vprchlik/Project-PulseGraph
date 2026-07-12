"""Chronos-2 forecasting backbone.

Wraps the chronos-forecasting library to provide probabilistic forecasts
with sampled trajectories for downstream regime analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from pulsegraph.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ForecastOutput:
    """Container for a single entity's forecast results."""

    repo_name: str
    horizon: int
    samples: np.ndarray          # shape: (n_samples, horizon)
    median: np.ndarray           # shape: (horizon,)
    mean: np.ndarray             # shape: (horizon,)
    lower_80: np.ndarray         # shape: (horizon,)
    upper_80: np.ndarray         # shape: (horizon,)
    lower_95: np.ndarray         # shape: (horizon,)
    upper_95: np.ndarray         # shape: (horizon,)
    context_length: int = 0
    model_name: str = ""
    metadata: dict = field(default_factory=dict)


class ChronosForecaster:
    """Probabilistic forecaster using Chronos-2 (amazon/chronos-2).

    Chronos-2 is a quantile-regression model: it emits quantile forecasts, not
    Monte-Carlo sample paths. To keep the evaluation metrics (empirical CRPS,
    PIT, coverage) identical in computation to the baselines, we request a dense
    grid of evenly-spaced quantile levels and treat those quantile values as a
    sample set (inverse-CDF / quantile sampling). With ~200 uniform levels this
    is a faithful discretization of the predictive distribution.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "auto",
        n_samples: int | None = None,
    ):
        self.model_name = model_name or "amazon/chronos-2"
        # Number of pseudo-samples reconstructed from the quantile forecast.
        self.n_samples = n_samples or settings.trajectory_samples
        self.device = device
        self._pipeline = None
        # The exact quantile levels Chronos-2 was trained on. Requesting levels
        # outside this set makes the model clamp to the extremes, creating tail
        # artifacts; instead we query these and interpolate the inverse CDF.
        self._native_levels = [
            0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
            0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99,
        ]
        # Uniform probability grid used to draw pseudo-samples via inverse-CDF.
        self._sample_probs = np.linspace(
            0.5 / self.n_samples,
            1 - 0.5 / self.n_samples,
            self.n_samples,
        )

    @property
    def pipeline(self):
        if self._pipeline is None:
            from chronos import Chronos2Pipeline

            device_map = self.device
            if device_map == "auto":
                device_map = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info("Loading Chronos-2 pipeline: %s on %s", self.model_name, device_map)
            self._pipeline = Chronos2Pipeline.from_pretrained(
                self.model_name,
                device_map=device_map,
            )
        return self._pipeline

    def _samples_from_quantiles(self, values: np.ndarray, horizon: int) -> np.ndarray:
        """Return pseudo-samples of shape (n_samples, horizon) from quantile forecast.

        Queries Chronos-2's native quantile levels, then reconstructs a dense
        sample set per horizon step by linearly interpolating the inverse CDF
        (piecewise-linear quantile function). Probabilities outside [0.01, 0.99]
        clamp to the endpoint quantiles, which is the model's honest tail estimate.
        """
        q, _ = self.pipeline.predict_quantiles(
            [values],
            prediction_length=horizon,
            quantile_levels=self._native_levels,
        )
        qa = np.asarray(q[0], dtype=np.float64)
        # Shape is (n_variates, horizon, n_levels) for univariate -> squeeze variate dim.
        if qa.ndim == 3:
            qa = qa[0]
        # qa: (horizon, n_native_levels). Enforce monotonicity then interpolate.
        qa = np.maximum.accumulate(qa, axis=1)
        horizon_len = qa.shape[0]
        samples = np.empty((self.n_samples, horizon_len), dtype=np.float64)
        for t in range(horizon_len):
            samples[:, t] = np.interp(self._sample_probs, self._native_levels, qa[t])
        return np.maximum(0.0, samples)

    def forecast(
        self,
        series: pd.Series | np.ndarray,
        horizon: int = 30,
        repo_name: str = "",
    ) -> ForecastOutput:
        """Generate a probabilistic forecast for a single time series."""
        if isinstance(series, pd.Series):
            values = series.values.astype(np.float32)
        else:
            values = np.asarray(series, dtype=np.float32)

        samples_np = self._samples_from_quantiles(values, horizon)

        return ForecastOutput(
            repo_name=repo_name,
            horizon=horizon,
            samples=samples_np,
            median=np.median(samples_np, axis=0),
            mean=np.mean(samples_np, axis=0),
            lower_80=np.percentile(samples_np, 10, axis=0),
            upper_80=np.percentile(samples_np, 90, axis=0),
            lower_95=np.percentile(samples_np, 2.5, axis=0),
            upper_95=np.percentile(samples_np, 97.5, axis=0),
            context_length=len(values),
            model_name="chronos2",
        )
