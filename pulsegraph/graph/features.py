"""Graph-derived features for use as forecasting covariates.

Computes node-level features: degree, PageRank, community ID,
dependency depth, and neighbor trajectory statistics.
"""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_graph_features(
    graph: nx.DiGraph,
    signal_dataset: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Compute graph-derived feature vectors for all repository nodes.

    Features:
    - in_degree, out_degree, total_degree
    - pagerank
    - dependency_depth (longest path in depends_on subgraph)
    - n_dep_edges, n_contributor_edges, n_org_edges, n_topic_edges
    - community_id (from undirected Louvain on unweighted projection)
    - neighbor_mean_stars, neighbor_std_stars (if signal data provided)
    """
    repo_nodes = [
        n for n, d in graph.nodes(data=True) if d.get("node_type") == "repository"
    ]
    if not repo_nodes:
        return pd.DataFrame()

    in_deg = dict(graph.in_degree(repo_nodes))
    out_deg = dict(graph.out_degree(repo_nodes))

    try:
        pr = nx.pagerank(graph, alpha=0.85, max_iter=100)
    except Exception:
        pr = {n: 0.0 for n in repo_nodes}

    dep_subgraph = nx.DiGraph()
    for u, v, d in graph.edges(data=True):
        if d.get("edge_type") == "depends_on":
            dep_subgraph.add_edge(u, v)

    dep_depth = {}
    for node in repo_nodes:
        if node in dep_subgraph:
            try:
                lengths = nx.single_source_shortest_path_length(dep_subgraph, node)
                dep_depth[node] = max(lengths.values()) if lengths else 0
            except Exception:
                dep_depth[node] = 0
        else:
            dep_depth[node] = 0

    edge_type_counts: dict[str, dict[str, int]] = {n: {} for n in repo_nodes}
    for u, v, d in graph.edges(data=True):
        et = d.get("edge_type", "unknown")
        if u in edge_type_counts:
            edge_type_counts[u][et] = edge_type_counts[u].get(et, 0) + 1

    community_map = _compute_communities(graph, repo_nodes)

    records = []
    for node in repo_nodes:
        etc = edge_type_counts.get(node, {})
        rec = {
            "repo_name": node,
            "in_degree": in_deg.get(node, 0),
            "out_degree": out_deg.get(node, 0),
            "total_degree": in_deg.get(node, 0) + out_deg.get(node, 0),
            "pagerank": pr.get(node, 0.0),
            "dependency_depth": dep_depth.get(node, 0),
            "n_dep_edges": etc.get("depends_on", 0),
            "n_contributor_edges": etc.get("co_contributor", 0),
            "n_org_edges": etc.get("same_org", 0),
            "n_topic_edges": etc.get("same_topic", 0),
            "community_id": community_map.get(node, -1),
        }

        if signal_dataset and node in signal_dataset:
            neighbor_vals = _neighbor_signal_stats(graph, node, signal_dataset)
            rec.update(neighbor_vals)

        records.append(rec)

    return pd.DataFrame(records)


def _compute_communities(graph: nx.DiGraph, repo_nodes: list[str]) -> dict[str, int]:
    """Compute community assignments using greedy modularity on undirected projection."""
    undirected = graph.to_undirected()
    repo_subgraph = undirected.subgraph(repo_nodes).copy()

    if repo_subgraph.number_of_nodes() == 0:
        return {}

    try:
        from networkx.algorithms.community import greedy_modularity_communities

        communities = greedy_modularity_communities(repo_subgraph)
        community_map = {}
        for idx, comm in enumerate(communities):
            for node in comm:
                community_map[node] = idx
        return community_map
    except Exception as e:
        logger.warning("Community detection failed: %s", e)
        return {n: 0 for n in repo_nodes}


def _neighbor_signal_stats(
    graph: nx.DiGraph,
    node: str,
    signal_dataset: dict[str, pd.Series],
) -> dict[str, float]:
    """Compute statistics of neighbor time series for a node."""
    neighbor_means = []
    neighbor_trends = []

    for _, target, _ in graph.edges(node, data=True):
        if target in signal_dataset:
            s = signal_dataset[target]
            neighbor_means.append(s.mean())
            if len(s) >= 7:
                trend = np.polyfit(range(min(30, len(s))), s.values[-min(30, len(s)):], 1)[0]
                neighbor_trends.append(trend)

    return {
        "neighbor_count": len(neighbor_means),
        "neighbor_mean_signal": float(np.mean(neighbor_means)) if neighbor_means else 0.0,
        "neighbor_std_signal": float(np.std(neighbor_means)) if neighbor_means else 0.0,
        "neighbor_mean_trend": float(np.mean(neighbor_trends)) if neighbor_trends else 0.0,
    }


def get_neighbor_series(
    graph: nx.DiGraph,
    repo_name: str,
    signal_dataset: dict[str, pd.Series],
    max_neighbors: int = 10,
    edge_types: list[str] | None = None,
) -> list[pd.Series]:
    """Retrieve time series for a repo's graph neighbors.

    Used as additional covariates for the forecasting model.
    """
    neighbor_series = []
    for _, target, data in graph.edges(repo_name, data=True):
        if edge_types and data.get("edge_type") not in edge_types:
            continue
        if target in signal_dataset:
            neighbor_series.append(signal_dataset[target])
            if len(neighbor_series) >= max_neighbors:
                break

    return neighbor_series
