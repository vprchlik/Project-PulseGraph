"""Trajectory sampling and regime analysis pipeline.

Coordinates the flow from forecast output -> trajectory samples ->
regime classification -> regime probability estimation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from pulsegraph.forecast.chronos_forecaster import ForecastOutput
from pulsegraph.regime.classifier import Regime, classify_trajectory_samples
from pulsegraph.regime.clustering import cluster_and_classify

logger = logging.getLogger(__name__)


@dataclass
class RegimeAnalysis:
    """Complete regime analysis for a single entity."""

    repo_name: str
    horizon: int
    regime_probs: dict[Regime, float]
    dominant_regime: Regime
    dominant_prob: float
    method: str = "shape_features"
    samples_used: int = 0
    metadata: dict = field(default_factory=dict)


def analyze_regimes(
    forecast: ForecastOutput,
    method: str = "shape_features",
    n_clusters: int = 5,
) -> RegimeAnalysis:
    """Analyze forecast samples to produce regime probabilities.

    Args:
        forecast: ForecastOutput with samples array
        method: "shape_features" (rule-based) or "dtw_clustering"
        n_clusters: for DTW clustering method
    """
    if method == "shape_features":
        regime_probs = classify_trajectory_samples(forecast.samples)
    elif method == "dtw_clustering":
        regime_probs = cluster_and_classify(forecast.samples, n_clusters)
    else:
        raise ValueError(f"Unknown method: {method}")

    dominant = max(regime_probs, key=regime_probs.get)

    return RegimeAnalysis(
        repo_name=forecast.repo_name,
        horizon=forecast.horizon,
        regime_probs=regime_probs,
        dominant_regime=dominant,
        dominant_prob=regime_probs[dominant],
        method=method,
        samples_used=forecast.samples.shape[0],
    )


def batch_regime_analysis(
    forecasts: list[ForecastOutput],
    method: str = "shape_features",
) -> list[RegimeAnalysis]:
    """Analyze regimes for a batch of forecasts."""
    results = []
    for f in forecasts:
        try:
            analysis = analyze_regimes(f, method=method)
            results.append(analysis)
        except Exception as e:
            logger.warning("Regime analysis failed for %s: %s", f.repo_name, e)
    return results
