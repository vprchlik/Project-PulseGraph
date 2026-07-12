"""TimesFM-2.5 forecasting backbone.

Wraps google/timesfm-2.5-200m-pytorch to produce probabilistic forecasts in the
same ForecastOutput format as the baselines and Chronos-2, so the evaluation
harness (empirical CRPS, PIT, coverage) is identical across all models.

Two important implementation notes:

1. Thread thrashing. On many-core CPUs, the default torch thread pool (one
   thread per core) causes severe contention and the model appears to hang for
   hours. We pin the thread count to a small number, after which load+forecast
   runs in a few seconds on CPU.

2. Quantile head. TimesFM-2.5 emits a fixed set of decile quantiles
   [0.1..0.9] (index 0 of the size-10 output is the mean). Like the Chronos-2
   wrapper, we reconstruct a dense pseudo-sample set per horizon step by
   inverse-CDF interpolation of those quantiles. Because TimesFM only exposes
   0.1..0.9 (vs Chronos's 0.01..0.99), its tails are coarser: probabilities
   outside [0.1, 0.9] cannot be identified. The wrapper therefore reports a
   supported 80% interval and marks the unsupported 95% interval as NaN rather
   than relabeling q0.1--q0.9 as q0.025--q0.975.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from pulsegraph.config import settings
from pulsegraph.forecast.chronos_forecaster import ForecastOutput

logger = logging.getLogger(__name__)

_TIMESFM_THREADS = int(os.environ.get("TIMESFM_THREADS", "4"))


class TimesFMForecaster:
    """Probabilistic forecaster using TimesFM-2.5 (200M, pytorch)."""

    def __init__(
        self,
        model_name: str = "google/timesfm-2.5-200m-pytorch",
        n_samples: int | None = None,
        max_context: int = 1024,
        max_horizon: int = 128,
    ):
        self.model_name = model_name
        self.n_samples = n_samples or settings.trajectory_samples
        self.max_context = max_context
        self.max_horizon = max_horizon
        self._model = None
        # TimesFM native decile levels (0.1..0.9). Index 0 of the output is the
        # mean; indices 1..9 are these quantiles.
        self._native_levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        self._sample_probs = np.linspace(
            0.5 / self.n_samples, 1 - 0.5 / self.n_samples, self.n_samples
        )

    @property
    def model(self):
        if self._model is None:
            import torch
            torch.set_num_threads(_TIMESFM_THREADS)
            import timesfm

            logger.info("Loading TimesFM-2.5 (threads=%d): %s",
                        _TIMESFM_THREADS, self.model_name)
            m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.model_name)
            m.compile(
                timesfm.ForecastConfig(
                    max_context=self.max_context,
                    max_horizon=self.max_horizon,
                    normalize_inputs=True,
                    use_continuous_quantile_head=True,
                    fix_quantile_crossing=True,
                )
            )
            self._model = m
        return self._model

    def _samples_from_quantiles(self, values: np.ndarray, horizon: int) -> np.ndarray:
        """Return pseudo-samples (n_samples, horizon) from TimesFM's deciles."""
        ctx = np.asarray(values, dtype=np.float32)
        _, q = self.model.forecast(horizon=horizon, inputs=[ctx])
        qa = np.asarray(q[0], dtype=np.float64)  # (horizon, 10)
        # Drop the mean (index 0); keep the 9 deciles.
        deciles = qa[:, 1:10]
        # Enforce monotonicity across quantile levels.
        deciles = np.maximum.accumulate(deciles, axis=1)
        horizon_len = deciles.shape[0]
        samples = np.empty((self.n_samples, horizon_len), dtype=np.float64)
        for t in range(horizon_len):
            samples[:, t] = np.interp(self._sample_probs, self._native_levels, deciles[t])
        return np.maximum(0.0, samples)

    def forecast(
        self,
        series: pd.Series | np.ndarray,
        horizon: int = 30,
        repo_name: str = "",
    ) -> ForecastOutput:
        if isinstance(series, pd.Series):
            values = series.values.astype(np.float32)
        else:
            values = np.asarray(series, dtype=np.float32)

        samples_np = self._samples_from_quantiles(values, horizon)
        unsupported_95 = np.full(horizon, np.nan, dtype=np.float64)

        return ForecastOutput(
            repo_name=repo_name,
            horizon=horizon,
            samples=samples_np,
            median=np.median(samples_np, axis=0),
            mean=np.mean(samples_np, axis=0),
            lower_80=np.percentile(samples_np, 10, axis=0),
            upper_80=np.percentile(samples_np, 90, axis=0),
            lower_95=unsupported_95.copy(),
            upper_95=unsupported_95.copy(),
            context_length=len(values),
            model_name="timesfm",
            metadata={
                "native_quantile_levels": self._native_levels.tolist(),
                "supported_interval_levels": [0.8],
            },
        )
