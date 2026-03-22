"""Tests for evaluation metrics."""

import numpy as np
import pytest

from pulsegraph.evaluation.metrics import (
    coverage,
    crps_empirical,
    mae,
    mape,
    pit_values,
)


def test_crps_perfect_forecast():
    samples = np.full(100, 5.0)
    assert crps_empirical(samples, 5.0) == pytest.approx(0.0, abs=1e-6)


def test_crps_wide_distribution():
    rng = np.random.default_rng(42)
    samples = rng.normal(0, 10, size=1000)
    score = crps_empirical(samples, 0.0)
    assert score > 0
    assert score < 10


def test_mape_basic():
    forecast = np.array([10.0, 20.0, 30.0])
    actuals = np.array([10.0, 25.0, 30.0])
    result = mape(forecast, actuals)
    assert 0 < result < 100


def test_mape_zero_actuals():
    forecast = np.array([1.0, 2.0])
    actuals = np.array([0.0, 0.0])
    result = mape(forecast, actuals)
    assert np.isnan(result)


def test_mae_basic():
    forecast = np.array([10.0, 20.0])
    actuals = np.array([12.0, 18.0])
    assert mae(forecast, actuals) == pytest.approx(2.0)


def test_coverage_perfect():
    lower = np.array([0.0, 0.0])
    upper = np.array([10.0, 10.0])
    actuals = np.array([5.0, 5.0])
    assert coverage(lower, upper, actuals) == pytest.approx(1.0)


def test_coverage_none_inside():
    lower = np.array([10.0, 10.0])
    upper = np.array([20.0, 20.0])
    actuals = np.array([5.0, 5.0])
    assert coverage(lower, upper, actuals) == pytest.approx(0.0)


def test_pit_uniform_calibration():
    rng = np.random.default_rng(42)
    n_steps = 50
    samples = rng.normal(0, 1, size=(1000, n_steps))
    actuals = rng.normal(0, 1, size=n_steps)
    pit = pit_values(samples, actuals)
    assert pit.shape == (n_steps,)
    assert np.all(pit >= 0) and np.all(pit <= 1)
