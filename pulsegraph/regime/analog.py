"""Analog retrieval: find historically similar entity-states.

Represents each entity's current state as a feature vector (trajectory shape,
signal levels, graph features, regime distribution) and finds nearest
neighbors among historical entity-states.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pulsegraph.regime.classifier import Regime, compute_shape_features

logger = logging.getLogger(__name__)


@dataclass
class EntityState:
    """Snapshot of an entity's state at a point in time."""

    repo_name: str
    state_date: str
    trajectory_features: np.ndarray
    regime_probs: dict[Regime, float] | None = None
    graph_features: dict[str, float] | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class AnalogMatch:
    """A historical analog match."""

    query_repo: str
    match_repo: str
    match_date: str
    similarity: float
    realized_outcome: str = ""
    metadata: dict = field(default_factory=dict)


class AnalogIndex:
    """Index of historical entity-states for analog retrieval."""

    def __init__(self):
        self.states: list[EntityState] = []
        self._vectors: np.ndarray | None = None
        self._dirty = True

    def add_state(self, state: EntityState):
        self.states.append(state)
        self._dirty = True

    def add_states_from_series(
        self,
        repo_name: str,
        series: pd.Series,
        window_size: int = 30,
        step: int = 7,
    ):
        """Build historical entity-states by sliding a window over the series."""
        values = series.values.astype(np.float64)
        n = len(values)

        for start in range(0, n - window_size, step):
            window = values[start : start + window_size]
            features = compute_shape_features(window)
            feature_vec = np.array([
                features.slope,
                features.curvature,
                features.volatility,
                features.spike_ratio,
                features.max_position,
                features.total_change,
                float(np.mean(window)),
                float(np.std(window)),
            ])

            date_str = ""
            if hasattr(series.index, "dtype") and pd.api.types.is_datetime64_any_dtype(
                series.index
            ):
                date_str = str(series.index[start + window_size - 1])

            self.add_state(EntityState(
                repo_name=repo_name,
                state_date=date_str,
                trajectory_features=feature_vec,
            ))

        self._dirty = True

    def _build_index(self):
        """Build the vector matrix for nearest-neighbor search."""
        if not self.states:
            self._vectors = np.empty((0, 0))
            return

        self._vectors = np.vstack([s.trajectory_features for s in self.states])

        mean = self._vectors.mean(axis=0)
        std = self._vectors.std(axis=0)
        std[std == 0] = 1.0
        self._vectors = (self._vectors - mean) / std
        self._norm_mean = mean
        self._norm_std = std

        self._dirty = False

    def query(
        self,
        query_state: EntityState,
        k: int = 10,
        exclude_same_repo: bool = True,
    ) -> list[AnalogMatch]:
        """Find the k nearest historical analogs to a query state."""
        if self._dirty:
            self._build_index()

        if self._vectors is None or self._vectors.shape[0] == 0:
            return []

        query_vec = (query_state.trajectory_features - self._norm_mean) / self._norm_std
        distances = np.linalg.norm(self._vectors - query_vec, axis=1)

        if exclude_same_repo:
            for i, state in enumerate(self.states):
                if state.repo_name == query_state.repo_name:
                    distances[i] = np.inf

        top_k = np.argsort(distances)[:k]

        matches = []
        for idx in top_k:
            if distances[idx] == np.inf:
                continue
            state = self.states[idx]
            similarity = 1.0 / (1.0 + distances[idx])
            matches.append(AnalogMatch(
                query_repo=query_state.repo_name,
                match_repo=state.repo_name,
                match_date=state.state_date,
                similarity=float(similarity),
            ))

        return matches

    def query_from_series(
        self,
        repo_name: str,
        series: pd.Series,
        window_size: int = 30,
        k: int = 10,
    ) -> list[AnalogMatch]:
        """Convenience: build a query state from the tail of a series and search."""
        values = series.values[-window_size:].astype(np.float64)
        features = compute_shape_features(values)
        feature_vec = np.array([
            features.slope,
            features.curvature,
            features.volatility,
            features.spike_ratio,
            features.max_position,
            features.total_change,
            float(np.mean(values)),
            float(np.std(values)),
        ])

        query = EntityState(
            repo_name=repo_name,
            state_date=str(series.index[-1]) if len(series) > 0 else "",
            trajectory_features=feature_vec,
        )
        return self.query(query, k=k)
