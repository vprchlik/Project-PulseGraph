"""Tests for regime classification."""

import numpy as np
import pytest

from pulsegraph.regime.classifier import (
    Regime,
    classify_trajectory,
    classify_trajectory_samples,
    compute_shape_features,
)


def test_breakout_trajectory():
    t = np.arange(30, dtype=float)
    trajectory = t ** 1.5
    regime = classify_trajectory(trajectory)
    assert regime in (Regime.BREAKOUT, Regime.STEADY_GROWTH)


def test_decay_trajectory():
    trajectory = np.linspace(100, 20, 30)
    regime = classify_trajectory(trajectory)
    assert regime == Regime.DECAY


def test_plateau_trajectory():
    trajectory = np.full(30, 50.0) + np.random.default_rng(42).normal(0, 0.5, 30)
    regime = classify_trajectory(trajectory)
    assert regime == Regime.PLATEAU


def test_spike_and_fade():
    trajectory = np.zeros(30)
    trajectory[:5] = [10, 50, 100, 80, 40]
    trajectory[5:] = np.linspace(20, 5, 25)
    regime = classify_trajectory(trajectory)
    assert regime == Regime.SPIKE_AND_FADE


def test_shape_features_nonzero():
    trajectory = np.arange(30, dtype=float)
    features = compute_shape_features(trajectory)
    assert features.slope > 0
    assert features.length == 30


def test_classify_samples_probabilities():
    rng = np.random.default_rng(42)
    samples = rng.normal(0, 1, size=(100, 30)).cumsum(axis=1)
    probs = classify_trajectory_samples(samples)
    assert sum(probs.values()) == pytest.approx(1.0)
    assert all(0 <= p <= 1 for p in probs.values())
