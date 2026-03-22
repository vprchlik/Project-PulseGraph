"""DTW-based trajectory clustering for regime discovery.

Provides an alternative to rule-based classification: cluster sampled
trajectories using Dynamic Time Warping distance, then label each cluster
using the shape-feature classifier.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering

from pulsegraph.regime.classifier import Regime, classify_trajectory

logger = logging.getLogger(__name__)


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute DTW distance between two 1-D sequences."""
    n, m = len(a), len(b)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = abs(float(a[i - 1]) - float(b[j - 1]))
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    return float(cost[n, m])


def _pairwise_dtw(samples: np.ndarray, max_samples: int = 100) -> np.ndarray:
    """Compute pairwise DTW distance matrix, subsampling if needed."""
    n = min(samples.shape[0], max_samples)
    if n < samples.shape[0]:
        indices = np.random.default_rng(42).choice(samples.shape[0], n, replace=False)
        samples = samples[indices]

    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = dtw_distance(samples[i], samples[j])
            dist[i, j] = d
            dist[j, i] = d

    return dist


def cluster_trajectories(
    samples: np.ndarray,
    n_clusters: int = 5,
    max_samples_for_dtw: int = 100,
) -> tuple[np.ndarray, dict[int, Regime]]:
    """Cluster sampled trajectories and label each cluster.

    Args:
        samples: shape (n_samples, horizon)
        n_clusters: target number of clusters
        max_samples_for_dtw: subsample for DTW computation

    Returns:
        (labels, cluster_regime_map)
        - labels: cluster assignment for each sample
        - cluster_regime_map: cluster_id -> Regime label
    """
    n_actual = min(samples.shape[0], max_samples_for_dtw)
    if n_actual < n_clusters:
        n_clusters = max(2, n_actual)

    dist_matrix = _pairwise_dtw(samples, max_samples=max_samples_for_dtw)

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average",
    )
    labels = clustering.fit_predict(dist_matrix)

    cluster_regime_map = {}
    for cid in range(n_clusters):
        mask = labels == cid
        if not np.any(mask):
            cluster_regime_map[cid] = Regime.PLATEAU
            continue

        cluster_samples = samples[:max_samples_for_dtw][mask]
        medoid_idx = np.argmin(dist_matrix[mask][:, mask].sum(axis=1))
        medoid = cluster_samples[medoid_idx]
        cluster_regime_map[cid] = classify_trajectory(medoid)

    if samples.shape[0] > max_samples_for_dtw:
        full_labels = _assign_remaining(samples, labels, max_samples_for_dtw)
    else:
        full_labels = labels

    return full_labels, cluster_regime_map


def _assign_remaining(
    samples: np.ndarray,
    partial_labels: np.ndarray,
    n_assigned: int,
) -> np.ndarray:
    """Assign remaining samples to nearest cluster centroid (Euclidean)."""
    centroids = {}
    for cid in np.unique(partial_labels):
        mask = partial_labels == cid
        centroids[cid] = samples[:n_assigned][mask].mean(axis=0)

    full_labels = np.zeros(samples.shape[0], dtype=int)
    full_labels[:n_assigned] = partial_labels

    centroid_arr = np.array([centroids[cid] for cid in sorted(centroids)])
    cid_order = sorted(centroids.keys())

    for i in range(n_assigned, samples.shape[0]):
        dists = np.linalg.norm(centroid_arr - samples[i], axis=1)
        full_labels[i] = cid_order[np.argmin(dists)]

    return full_labels


def cluster_and_classify(
    samples: np.ndarray,
    n_clusters: int = 5,
) -> dict[Regime, float]:
    """Full pipeline: cluster trajectories, label clusters, return regime probabilities."""
    labels, cluster_regime_map = cluster_trajectories(samples, n_clusters)

    regime_counts = {r: 0 for r in Regime}
    for label in labels:
        regime = cluster_regime_map.get(label, Regime.PLATEAU)
        regime_counts[regime] += 1

    total = len(labels)
    return {r: count / total for r, count in regime_counts.items()}
