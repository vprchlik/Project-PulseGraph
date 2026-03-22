"""Tests for analog retrieval."""

import numpy as np
import pandas as pd
import pytest

from pulsegraph.regime.analog import AnalogIndex


@pytest.fixture
def analog_index():
    rng = np.random.default_rng(42)
    index = AnalogIndex()

    dates = pd.date_range("2022-01-01", periods=365, freq="D")
    for i in range(20):
        values = np.maximum(0, 10 + 0.1 * np.arange(365) + rng.normal(0, 3, 365))
        series = pd.Series(values, index=dates)
        index.add_states_from_series(f"org/repo-{i}", series, window_size=30, step=14)

    return index


def test_analog_index_populated(analog_index):
    assert len(analog_index.states) > 0


def test_analog_query_returns_matches(analog_index):
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    query_series = pd.Series(np.arange(60, dtype=float), index=dates)
    matches = analog_index.query_from_series("new/repo", query_series, k=5)
    assert len(matches) <= 5
    assert all(m.similarity > 0 for m in matches)


def test_analog_excludes_same_repo(analog_index):
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    query_series = pd.Series(np.arange(60, dtype=float), index=dates)
    matches = analog_index.query_from_series("org/repo-0", query_series, k=5)
    for m in matches:
        assert m.match_repo != "org/repo-0"
