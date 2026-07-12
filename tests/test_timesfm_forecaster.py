"""Unit tests for the TimesFM quantile adapter without loading model weights."""

import numpy as np

from pulsegraph.forecast.timesfm_forecaster import TimesFMForecaster


class _FakeTimesFM:
    def forecast(self, horizon, inputs):
        mean = np.full((1, horizon), 5.0)
        # Output index 0 is mean; indices 1..9 are q0.1..q0.9.
        quantiles = np.tile(np.arange(1.0, 11.0), (1, horizon, 1))
        return mean, quantiles


def test_timesfm_marks_95_interval_unsupported():
    forecaster = TimesFMForecaster(n_samples=20)
    forecaster._model = _FakeTimesFM()
    result = forecaster.forecast(np.arange(30.0), horizon=7, repo_name="test/repo")
    assert np.all(np.isfinite(result.lower_80))
    assert np.all(np.isfinite(result.upper_80))
    assert np.all(np.isnan(result.lower_95))
    assert np.all(np.isnan(result.upper_95))
    assert result.metadata["supported_interval_levels"] == [0.8]

