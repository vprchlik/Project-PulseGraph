"""Baseline forecasting models for comparison.

Implements four baselines from the evaluation protocol:
1. Naive seasonal (repeat last N days)
2. Linear extrapolation
3. Exponential smoothing (ETS)
4. LightGBM with handcrafted features
"""

from __future__ import annotations

import hashlib
import logging

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from pulsegraph.forecast.chronos_forecaster import ForecastOutput

logger = logging.getLogger(__name__)


def _forecast_rng(
    model_name: str,
    repo_name: str,
    horizon: int,
    values: np.ndarray,
) -> np.random.Generator:
    """Build a stable per-forecast RNG for reproducible bootstrap samples."""
    tail = np.asarray(values[-32:], dtype=np.float64)
    payload = (
        f"{model_name}|{repo_name}|{horizon}|{len(values)}|".encode()
        + tail.tobytes()
    )
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(seed)


def _bootstrap_samples(
    point_forecast: np.ndarray,
    residuals: np.ndarray,
    n_samples: int = 200,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate sample trajectories by adding bootstrapped residuals."""
    if rng is None:
        rng = np.random.default_rng()
    horizon = len(point_forecast)
    noise = rng.choice(residuals, size=(n_samples, horizon), replace=True)
    samples = point_forecast[np.newaxis, :] + noise
    return np.maximum(0, samples)


class NaiveSeasonalForecaster:
    """Repeat the last `period` days of the series."""

    def __init__(self, n_samples: int = 200):
        self.n_samples = n_samples

    def forecast(
        self,
        series: pd.Series | np.ndarray,
        horizon: int = 30,
        repo_name: str = "",
    ) -> ForecastOutput:
        values = np.asarray(series, dtype=np.float64)
        period = min(horizon, len(values))
        pattern = values[-period:]

        reps = (horizon // period) + 1
        point = np.tile(pattern, reps)[:horizon]

        residuals = np.diff(values[-min(90, len(values)):])
        if len(residuals) == 0:
            residuals = np.array([0.0])
        samples = _bootstrap_samples(
            point,
            residuals,
            self.n_samples,
            rng=_forecast_rng("naive_seasonal", repo_name, horizon, values),
        )

        return ForecastOutput(
            repo_name=repo_name,
            horizon=horizon,
            samples=samples,
            median=np.median(samples, axis=0),
            mean=np.mean(samples, axis=0),
            lower_80=np.percentile(samples, 10, axis=0),
            upper_80=np.percentile(samples, 90, axis=0),
            lower_95=np.percentile(samples, 2.5, axis=0),
            upper_95=np.percentile(samples, 97.5, axis=0),
            context_length=len(values),
            model_name="naive_seasonal",
        )


class LinearForecaster:
    """Fit a line to the last `lookback` days, project forward."""

    def __init__(self, lookback: int = 30, n_samples: int = 200):
        self.lookback = lookback
        self.n_samples = n_samples

    def forecast(
        self,
        series: pd.Series | np.ndarray,
        horizon: int = 30,
        repo_name: str = "",
    ) -> ForecastOutput:
        values = np.asarray(series, dtype=np.float64)
        tail = values[-self.lookback:]
        x = np.arange(len(tail))

        slope, intercept, _, _, stderr = stats.linregress(x, tail)
        future_x = np.arange(len(tail), len(tail) + horizon)
        point = slope * future_x + intercept
        point = np.maximum(0, point)

        residuals = tail - (slope * x + intercept)
        samples = _bootstrap_samples(
            point,
            residuals,
            self.n_samples,
            rng=_forecast_rng("linear", repo_name, horizon, values),
        )

        return ForecastOutput(
            repo_name=repo_name,
            horizon=horizon,
            samples=samples,
            median=np.median(samples, axis=0),
            mean=np.mean(samples, axis=0),
            lower_80=np.percentile(samples, 10, axis=0),
            upper_80=np.percentile(samples, 90, axis=0),
            lower_95=np.percentile(samples, 2.5, axis=0),
            upper_95=np.percentile(samples, 97.5, axis=0),
            context_length=len(values),
            model_name="linear",
        )


class ETSForecaster:
    """Exponential smoothing (Holt-Winters) forecaster."""

    def __init__(self, n_samples: int = 200):
        self.n_samples = n_samples

    def forecast(
        self,
        series: pd.Series | np.ndarray,
        horizon: int = 30,
        repo_name: str = "",
    ) -> ForecastOutput:
        values = np.asarray(series, dtype=np.float64)
        values_safe = np.maximum(values, 0) + 0.01

        try:
            model = ExponentialSmoothing(
                values_safe,
                trend="add",
                damped_trend=True,
                seasonal=None,
                initialization_method="estimated",
            ).fit(optimized=True)
            point = model.forecast(horizon)
            point = np.maximum(0, point)

            fitted = model.fittedvalues
            residuals = values_safe[: len(fitted)] - fitted
        except Exception as e:
            logger.warning("ETS fit failed for %s: %s, falling back to mean", repo_name, e)
            mean_val = np.mean(values[-30:])
            point = np.full(horizon, mean_val)
            residuals = values[-30:] - mean_val

        samples = _bootstrap_samples(
            point,
            residuals,
            self.n_samples,
            rng=_forecast_rng("ets", repo_name, horizon, values),
        )

        return ForecastOutput(
            repo_name=repo_name,
            horizon=horizon,
            samples=samples,
            median=np.median(samples, axis=0),
            mean=np.mean(samples, axis=0),
            lower_80=np.percentile(samples, 10, axis=0),
            upper_80=np.percentile(samples, 90, axis=0),
            lower_95=np.percentile(samples, 2.5, axis=0),
            upper_95=np.percentile(samples, 97.5, axis=0),
            context_length=len(values),
            model_name="ets",
        )


class LightGBMForecaster:
    """LightGBM with handcrafted time-series features."""

    def __init__(self, n_samples: int = 200):
        self.n_samples = n_samples
        self._model = None

    @staticmethod
    def _build_features(values: np.ndarray, idx: int) -> dict:
        """Build feature vector for predicting values[idx] from history."""
        features = {}
        for lag in [1, 2, 3, 7, 14, 30]:
            pos = idx - lag
            features[f"lag_{lag}"] = values[pos] if pos >= 0 else 0.0

        for window in [7, 14, 30]:
            start = max(0, idx - window)
            segment = values[start:idx] if idx > start else np.array([0.0])
            features[f"mean_{window}"] = np.mean(segment)
            features[f"std_{window}"] = np.std(segment)
            features[f"max_{window}"] = np.max(segment)
            features[f"min_{window}"] = np.min(segment)

        if idx >= 7:
            features["trend_7"] = float(np.polyfit(range(7), values[idx - 7 : idx], 1)[0])
        else:
            features["trend_7"] = 0.0

        features["day_of_week"] = idx % 7
        return features

    def forecast(
        self,
        series: pd.Series | np.ndarray,
        horizon: int = 30,
        repo_name: str = "",
    ) -> ForecastOutput:
        import lightgbm as lgb

        values = np.asarray(series, dtype=np.float64)

        train_start = max(60, len(values) // 2)
        X_train, y_train = [], []
        for i in range(train_start, len(values)):
            X_train.append(self._build_features(values, i))
            y_train.append(values[i])

        X_df = pd.DataFrame(X_train)
        y_arr = np.array(y_train)

        model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            min_child_samples=5,
            verbose=-1,
        )
        model.fit(X_df, y_arr)

        residuals = y_arr - model.predict(X_df)

        extended = np.concatenate([values, np.zeros(horizon)])
        for step in range(horizon):
            idx = len(values) + step
            feat = self._build_features(extended, idx)
            pred = model.predict(pd.DataFrame([feat]))[0]
            extended[idx] = max(0, pred)

        point = extended[len(values):]
        samples = _bootstrap_samples(
            point,
            residuals,
            self.n_samples,
            rng=_forecast_rng("lightgbm", repo_name, horizon, values),
        )

        return ForecastOutput(
            repo_name=repo_name,
            horizon=horizon,
            samples=samples,
            median=np.median(samples, axis=0),
            mean=np.mean(samples, axis=0),
            lower_80=np.percentile(samples, 10, axis=0),
            upper_80=np.percentile(samples, 90, axis=0),
            lower_95=np.percentile(samples, 2.5, axis=0),
            upper_95=np.percentile(samples, 97.5, axis=0),
            context_length=len(values),
            model_name="lightgbm",
        )


class LastValueForecaster:
    """Random-walk baseline: repeat the most recent observation."""

    def __init__(self, n_samples: int = 200):
        self.n_samples = n_samples

    def forecast(
        self,
        series: pd.Series | np.ndarray,
        horizon: int = 30,
        repo_name: str = "",
    ) -> ForecastOutput:
        values = np.asarray(series, dtype=np.float64)
        point = np.full(horizon, values[-1])
        tail = values[-min(91, len(values)) :]
        residuals = np.diff(tail)
        if len(residuals) == 0:
            residuals = np.array([0.0])
        samples = _bootstrap_samples(
            point,
            residuals,
            self.n_samples,
            rng=_forecast_rng("last_value", repo_name, horizon, values),
        )
        return _forecast_output(repo_name, horizon, samples, len(values), "last_value")


class WeeklySeasonalNaiveForecaster:
    """Fixed weekly seasonal naive: repeat the last seven observations."""

    def __init__(self, n_samples: int = 200):
        self.n_samples = n_samples

    def forecast(
        self,
        series: pd.Series | np.ndarray,
        horizon: int = 30,
        repo_name: str = "",
    ) -> ForecastOutput:
        values = np.asarray(series, dtype=np.float64)
        period = min(7, len(values))
        point = np.resize(values[-period:], horizon)
        if len(values) > period:
            residuals = values[period:] - values[:-period]
            residuals = residuals[-min(90, len(residuals)) :]
        else:
            residuals = np.array([0.0])
        samples = _bootstrap_samples(
            point,
            residuals,
            self.n_samples,
            rng=_forecast_rng("weekly_naive", repo_name, horizon, values),
        )
        return _forecast_output(repo_name, horizon, samples, len(values), "weekly_naive")


class LocalMeanForecaster:
    """Robust local-level baseline using the recent 28-day mean."""

    def __init__(self, lookback: int = 28, n_samples: int = 200):
        self.lookback = lookback
        self.n_samples = n_samples

    def forecast(
        self,
        series: pd.Series | np.ndarray,
        horizon: int = 30,
        repo_name: str = "",
    ) -> ForecastOutput:
        values = np.asarray(series, dtype=np.float64)
        tail = values[-min(self.lookback, len(values)) :]
        level = float(np.mean(tail))
        point = np.full(horizon, level)
        residuals = tail - level
        if len(residuals) == 0:
            residuals = np.array([0.0])
        samples = _bootstrap_samples(
            point,
            residuals,
            self.n_samples,
            rng=_forecast_rng("local_mean", repo_name, horizon, values),
        )
        return _forecast_output(repo_name, horizon, samples, len(values), "local_mean")


def _forecast_output(
    repo_name: str,
    horizon: int,
    samples: np.ndarray,
    context_length: int,
    model_name: str,
) -> ForecastOutput:
    """Construct a ForecastOutput shared by simple baseline implementations."""
    return ForecastOutput(
        repo_name=repo_name,
        horizon=horizon,
        samples=samples,
        median=np.median(samples, axis=0),
        mean=np.mean(samples, axis=0),
        lower_80=np.percentile(samples, 10, axis=0),
        upper_80=np.percentile(samples, 90, axis=0),
        lower_95=np.percentile(samples, 2.5, axis=0),
        upper_95=np.percentile(samples, 97.5, axis=0),
        context_length=context_length,
        model_name=model_name,
    )


BASELINE_FORECASTERS = {
    "naive_seasonal": NaiveSeasonalForecaster,
    "linear": LinearForecaster,
    "ets": ETSForecaster,
    "lightgbm": LightGBMForecaster,
    "last_value": LastValueForecaster,
    "weekly_naive": WeeklySeasonalNaiveForecaster,
    "local_mean": LocalMeanForecaster,
}
