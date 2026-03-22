"""Event attribution: find events temporally correlated with trajectory changes.

Retrieves events near significant trajectory changes and scores them
by temporal proximity and event magnitude.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pulsegraph.data.events import fetch_events_for_repo

logger = logging.getLogger(__name__)


def detect_changepoints(
    series: pd.Series,
    window: int = 7,
    threshold_std: float = 2.0,
) -> list[dict]:
    """Detect significant trajectory changes using rolling z-score."""
    values = series.values.astype(np.float64)
    if len(values) < window * 2:
        return []

    rolling_mean = pd.Series(values).rolling(window, min_periods=1).mean().values
    rolling_std = pd.Series(values).rolling(window, min_periods=1).std().values
    rolling_std[rolling_std < 1e-6] = 1.0

    changes = []
    for i in range(window, len(values)):
        z = abs(values[i] - rolling_mean[i - 1]) / rolling_std[i - 1]
        if z > threshold_std:
            date_str = ""
            if hasattr(series.index, "dtype") and pd.api.types.is_datetime64_any_dtype(
                series.index
            ):
                date_str = str(series.index[i].date())

            changes.append({
                "index": i,
                "date": date_str,
                "z_score": float(z),
                "value": float(values[i]),
                "expected": float(rolling_mean[i - 1]),
                "direction": "up" if values[i] > rolling_mean[i - 1] else "down",
            })

    return changes


def attribute_events(
    repo_name: str,
    series: pd.Series,
    max_events: int = 20,
) -> list[dict]:
    """Find events temporally correlated with trajectory changes.

    For each detected changepoint, retrieves events within a window and
    scores them by proximity.
    """
    changepoints = detect_changepoints(series)
    if not changepoints:
        return []

    try:
        events_df = fetch_events_for_repo(repo_name)
    except Exception as e:
        logger.warning("Failed to fetch events for %s: %s", repo_name, e)
        return []

    if events_df.empty:
        return []

    attributed = []
    for cp in changepoints:
        cp_date = cp.get("date", "")
        if not cp_date:
            continue

        cp_ts = pd.Timestamp(cp_date)

        for _, event in events_df.iterrows():
            ev_date = event.get("event_date")
            if ev_date is None:
                continue

            ev_ts = pd.Timestamp(ev_date)
            days_diff = abs((cp_ts - ev_ts).days)

            if days_diff <= 7:
                proximity_score = 1.0 / (1.0 + days_diff)
                z_weight = min(cp["z_score"] / 3.0, 1.0)
                relevance = proximity_score * z_weight

                attributed.append({
                    "event_type": event.get("event_type", ""),
                    "event_date": str(ev_date),
                    "title": event.get("title", ""),
                    "url": event.get("url", ""),
                    "source": event.get("source", ""),
                    "relevance_score": float(relevance),
                    "changepoint_date": cp_date,
                    "changepoint_direction": cp["direction"],
                    "changepoint_z": cp["z_score"],
                })

    attributed.sort(key=lambda x: x["relevance_score"], reverse=True)
    return attributed[:max_events]
