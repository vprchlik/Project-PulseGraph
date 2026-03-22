"""Shape-feature-based regime classification.

Classifies a trajectory into one of five regimes based on computed shape
descriptors: slope, curvature, volatility, and spike ratio.

Regimes:
- breakout: sustained acceleration (positive curvature + strong positive slope)
- steady_growth: positive linear trend
- plateau: near-zero slope
- spike_and_fade: high initial spike followed by decay
- decay: sustained negative trend
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Regime(str, Enum):
    BREAKOUT = "breakout"
    STEADY_GROWTH = "steady_growth"
    PLATEAU = "plateau"
    SPIKE_AND_FADE = "spike_and_fade"
    DECAY = "decay"


@dataclass
class ShapeFeatures:
    """Trajectory shape descriptors."""

    slope: float
    curvature: float
    volatility: float
    spike_ratio: float
    max_position: float
    total_change: float
    length: int


def compute_shape_features(trajectory: np.ndarray) -> ShapeFeatures:
    """Compute shape features from a single trajectory."""
    n = len(trajectory)
    if n < 2:
        return ShapeFeatures(0, 0, 0, 0, 0, 0, n)

    t = np.arange(n, dtype=np.float64)
    vals = np.asarray(trajectory, dtype=np.float64)

    coeffs = np.polyfit(t, vals, 1)
    slope = coeffs[0]

    if n >= 3:
        coeffs2 = np.polyfit(t, vals, 2)
        curvature = coeffs2[0]
    else:
        curvature = 0.0

    residuals = vals - np.polyval(np.polyfit(t, vals, 1), t)
    volatility = float(np.std(residuals))

    val_range = np.max(vals) - np.min(vals)
    if val_range > 0:
        spike_ratio = (np.max(vals) - vals[-1]) / val_range
    else:
        spike_ratio = 0.0

    max_position = float(np.argmax(vals)) / max(n - 1, 1)
    total_change = float(vals[-1] - vals[0])

    return ShapeFeatures(
        slope=float(slope),
        curvature=float(curvature),
        volatility=float(volatility),
        spike_ratio=float(spike_ratio),
        max_position=float(max_position),
        total_change=float(total_change),
        length=n,
    )


def classify_trajectory(trajectory: np.ndarray) -> Regime:
    """Classify a single trajectory into a regime using shape features."""
    features = compute_shape_features(trajectory)

    scale = max(abs(features.total_change), np.std(trajectory), 1e-6)
    norm_slope = features.slope * features.length / scale
    norm_curvature = features.curvature * features.length**2 / scale

    if features.spike_ratio > 0.6 and features.max_position < 0.4:
        return Regime.SPIKE_AND_FADE

    if norm_curvature > 0.3 and norm_slope > 0.2:
        return Regime.BREAKOUT

    if norm_slope > 0.1:
        return Regime.STEADY_GROWTH

    if norm_slope < -0.1:
        return Regime.DECAY

    return Regime.PLATEAU


def classify_trajectory_samples(
    samples: np.ndarray,
) -> dict[Regime, float]:
    """Classify each sample trajectory and return regime probability distribution.

    Args:
        samples: shape (n_samples, horizon)

    Returns:
        Dict mapping each regime to its probability (fraction of samples).
    """
    n_samples = samples.shape[0]
    counts = {r: 0 for r in Regime}

    for i in range(n_samples):
        regime = classify_trajectory(samples[i])
        counts[regime] += 1

    return {r: count / n_samples for r, count in counts.items()}
