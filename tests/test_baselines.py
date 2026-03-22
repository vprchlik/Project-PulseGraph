"""Tests for baseline forecasters."""

import numpy as np
import pytest

from pulsegraph.forecast.baselines import (
    ETSForecaster,
    LinearForecaster,
    NaiveSeasonalForecaster,
)


@pytest.fixture
def sample_series():
    rng = np.random.default_rng(42)
    return np.maximum(0, 10 + 0.1 * np.arange(365) + rng.normal(0, 3, 365))


def test_naive_forecast_shape(sample_series):
    f = NaiveSeasonalForecaster(n_samples=50)
    result = f.forecast(sample_series, horizon=30, repo_name="test")
    assert result.samples.shape == (50, 30)
    assert result.median.shape == (30,)
    assert result.model_name == "naive_seasonal"


def test_linear_forecast_shape(sample_series):
    f = LinearForecaster(n_samples=50)
    result = f.forecast(sample_series, horizon=30, repo_name="test")
    assert result.samples.shape == (50, 30)
    assert result.median.shape == (30,)


def test_ets_forecast_shape(sample_series):
    f = ETSForecaster(n_samples=50)
    result = f.forecast(sample_series, horizon=30, repo_name="test")
    assert result.samples.shape == (50, 30)
    assert result.median.shape == (30,)


def test_forecast_nonnegative(sample_series):
    f = LinearForecaster(n_samples=100)
    result = f.forecast(sample_series, horizon=30)
    assert np.all(result.median >= 0)


def test_confidence_intervals_ordered(sample_series):
    f = ETSForecaster(n_samples=100)
    result = f.forecast(sample_series, horizon=30)
    assert np.all(result.lower_95 <= result.lower_80)
    assert np.all(result.lower_80 <= result.median)
    assert np.all(result.median <= result.upper_80)
    assert np.all(result.upper_80 <= result.upper_95)
