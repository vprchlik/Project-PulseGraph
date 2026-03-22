"""Forecast evaluation metrics.

Implements CRPS, MAPE, coverage, and calibration measures as specified
in the evaluation protocol.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats

from pulsegraph.forecast.chronos_forecaster import ForecastOutput


def crps_empirical(samples: np.ndarray, observation: float) -> float:
    """Compute the Continuous Ranked Probability Score from empirical samples.

    CRPS = E|X - y| - 0.5 * E|X - X'|
    where X, X' are iid draws from the forecast distribution.
    """
    samples = np.asarray(samples, dtype=np.float64)
    term1 = np.mean(np.abs(samples - observation))
    n = len(samples)
    if n > 1:
        sorted_s = np.sort(samples)
        diff_sum = 0.0
        for i in range(n):
            diff_sum += (2 * i + 1 - n) * sorted_s[i]
        term2 = diff_sum / (n * (n - 1))
    else:
        term2 = 0.0
    return term1 - term2


def crps_for_forecast(forecast: ForecastOutput, actuals: np.ndarray) -> np.ndarray:
    """Compute per-step CRPS across the forecast horizon."""
    horizon = min(forecast.horizon, len(actuals))
    crps_values = np.zeros(horizon)
    for t in range(horizon):
        crps_values[t] = crps_empirical(forecast.samples[:, t], actuals[t])
    return crps_values


def mape(forecast_median: np.ndarray, actuals: np.ndarray) -> float:
    """Mean Absolute Percentage Error, excluding zero actuals."""
    mask = actuals != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((actuals[mask] - forecast_median[mask]) / actuals[mask])) * 100)


def mae(forecast_median: np.ndarray, actuals: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(actuals - forecast_median)))


def coverage(
    lower: np.ndarray,
    upper: np.ndarray,
    actuals: np.ndarray,
) -> float:
    """Fraction of actuals falling within the prediction interval."""
    inside = (actuals >= lower) & (actuals <= upper)
    return float(np.mean(inside))


def pit_values(samples: np.ndarray, actuals: np.ndarray) -> np.ndarray:
    """Compute Probability Integral Transform values.

    For well-calibrated forecasts, PIT values should be Uniform(0, 1).
    """
    horizon = min(samples.shape[1], len(actuals))
    pit = np.zeros(horizon)
    for t in range(horizon):
        pit[t] = np.mean(samples[:, t] <= actuals[t])
    return pit


def ks_calibration_test(pit_vals: np.ndarray) -> tuple[float, float]:
    """Kolmogorov-Smirnov test for uniformity of PIT values.

    Returns (statistic, p_value). High p-value indicates good calibration.
    """
    stat, p = scipy_stats.kstest(pit_vals, "uniform")
    return float(stat), float(p)


def evaluate_forecast(
    forecast: ForecastOutput,
    actuals: np.ndarray,
) -> dict:
    """Compute all metrics for a single forecast vs actuals."""
    horizon = min(forecast.horizon, len(actuals))
    act = actuals[:horizon]
    med = forecast.median[:horizon]

    crps_vals = crps_for_forecast(forecast, act)
    pit_vals = pit_values(forecast.samples, act)

    cov_80 = coverage(
        forecast.lower_80[:horizon],
        forecast.upper_80[:horizon],
        act,
    )
    cov_95 = coverage(
        forecast.lower_95[:horizon],
        forecast.upper_95[:horizon],
        act,
    )

    ks_stat, ks_p = ks_calibration_test(pit_vals)

    return {
        "repo_name": forecast.repo_name,
        "model_name": forecast.model_name,
        "horizon": horizon,
        "crps_mean": float(np.mean(crps_vals)),
        "crps_median": float(np.median(crps_vals)),
        "mape": mape(med, act),
        "mae": mae(med, act),
        "coverage_80": cov_80,
        "coverage_95": cov_95,
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_p,
        "pit_values": pit_vals.tolist(),
    }
