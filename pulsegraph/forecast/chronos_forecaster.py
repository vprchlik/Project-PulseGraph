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
    """Probabilistic forecaster using Chronos-2."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "auto",
        n_samples: int | None = None,
    ):
        self.model_name = model_name or settings.chronos_model_name
        self.n_samples = n_samples or settings.trajectory_samples
        self.device = device
        self._pipeline = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            from chronos import ChronosPipeline

            device_map = self.device
            if device_map == "auto":
                device_map = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info("Loading Chronos pipeline: %s on %s", self.model_name, device_map)
            self._pipeline = ChronosPipeline.from_pretrained(
                self.model_name,
                device_map=device_map,
                torch_dtype=torch.float32,
            )
        return self._pipeline

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

        context = torch.tensor(values, dtype=torch.float32).unsqueeze(0)

        samples = self.pipeline.predict(
            context,
            prediction_length=horizon,
            num_samples=self.n_samples,
        )

        samples_np = samples.squeeze(0).numpy()

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
            model_name=self.model_name,
        )

    def forecast_batch(
        self,
        dataset: dict[str, pd.Series],
        horizon: int = 30,
    ) -> list[ForecastOutput]:
        """Forecast multiple series."""
        contexts = []
        names = []
        for name, series in dataset.items():
            values = series.values.astype(np.float32)
            contexts.append(torch.tensor(values, dtype=torch.float32))
            names.append(name)

        logger.info("Batch forecasting %d series, horizon=%d", len(contexts), horizon)
        all_samples = self.pipeline.predict(
            contexts,
            prediction_length=horizon,
            num_samples=self.n_samples,
        )

        results = []
        for i, name in enumerate(names):
            s = all_samples[i].numpy()
            results.append(ForecastOutput(
                repo_name=name,
                horizon=horizon,
                samples=s,
                median=np.median(s, axis=0),
                mean=np.mean(s, axis=0),
                lower_80=np.percentile(s, 10, axis=0),
                upper_80=np.percentile(s, 90, axis=0),
                lower_95=np.percentile(s, 2.5, axis=0),
                upper_95=np.percentile(s, 97.5, axis=0),
                context_length=len(dataset[name]),
                model_name=self.model_name,
            ))
        return results
