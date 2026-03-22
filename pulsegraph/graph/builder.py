"""Entity graph construction from multiple data sources.

Builds a heterogeneous NetworkX graph from:
- Libraries.io dependency edges
- GitHub contributor overlap
- Organization co-membership
- Topic co-occurrence
"""

from __future__ import annotations

import logging
from collections import defaultdict

import networkx as nx
import pandas as pd

from pulsegraph.data.github_api import GitHubAPIClient

logger = logging.getLogger(__name__)


class EntityGraphBuilder:
    """Incrementally builds a heterogeneous entity graph."""

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_repos(self, repos_df: pd.DataFrame):
        """Add repository nodes from a metadata DataFrame."""
        for _, row in repos_df.iterrows():
            name = row.get("repo_name", "")
            if not name:
                continue
            self.graph.add_node(
                name,
                node_type="repository",
                language=row.get("language", ""),
                license=row.get("license", ""),
                stratum=row.get("stratum", "unknown"),
                topics=row.get("topics", []),
                stargazers_count=row.get("stargazers_count", 0),
            )

    def add_dependency_edges(self, edges_df: pd.DataFrame):
        """Add depends_on edges from Libraries.io data."""
        added = 0
        for _, row in edges_df.iterrows():
            src = row.get("source_repo", "")
            tgt = row.get("target_repo", "")
            if src and tgt and src in self.graph and tgt in self.graph:
                self.graph.add_edge(src, tgt, edge_type="depends_on")
                added += 1
        logger.info("Added %d dependency edges", added)

    def add_contributor_edges(
        self,
        repo_names: list[str],
        gh_client: GitHubAPIClient | None = None,
        min_shared: int = 2,
    ):
        """Add co_contributor edges based on shared contributors.

        Two repos share an edge if they have >= min_shared contributors
        in common.
        """
        client = gh_client or GitHubAPIClient()

        repo_contributors: dict[str, set[str]] = {}
        for name in repo_names:
            if name not in self.graph:
                continue
            parts = name.split("/")
            if len(parts) != 2:
                continue
            try:
                contribs = client.get_repo_contributors(parts[0], parts[1], max_pages=2)
                repo_contributors[name] = {c["login"] for c in contribs if "login" in c}
            except Exception as e:
                logger.debug("Failed to get contributors for %s: %s", name, e)

        contributor_to_repos: dict[str, list[str]] = defaultdict(list)
        for repo, contribs in repo_contributors.items():
            for c in contribs:
                contributor_to_repos[c].append(repo)

        added = 0
        seen = set()
        for repos in contributor_to_repos.values():
            if len(repos) < 2:
                continue
            for i, r1 in enumerate(repos):
                for r2 in repos[i + 1 :]:
                    pair = tuple(sorted([r1, r2]))
                    if pair in seen:
                        continue
                    seen.add(pair)

                    shared = len(
                        repo_contributors.get(r1, set())
                        & repo_contributors.get(r2, set())
                    )
                    if shared >= min_shared:
                        self.graph.add_edge(
                            r1, r2, edge_type="co_contributor", weight=shared
                        )
                        self.graph.add_edge(
                            r2, r1, edge_type="co_contributor", weight=shared
                        )
                        added += 1
        logger.info("Added %d co-contributor edges", added)

    def add_org_edges(self):
        """Add same_org edges between repos in the same GitHub organization."""
        org_repos: dict[str, list[str]] = defaultdict(list)
        for node, data in self.graph.nodes(data=True):
            if data.get("node_type") != "repository":
                continue
            org = node.split("/")[0] if "/" in node else ""
            if org:
                org_repos[org].append(node)

        added = 0
        for repos in org_repos.values():
            if len(repos) < 2:
                continue
            for i, r1 in enumerate(repos):
                for r2 in repos[i + 1 :]:
                    self.graph.add_edge(r1, r2, edge_type="same_org")
                    self.graph.add_edge(r2, r1, edge_type="same_org")
                    added += 1
        logger.info("Added %d same-org edges", added)

    def add_topic_edges(self, min_shared_topics: int = 2):
        """Add same_topic edges between repos sharing topics."""
        topic_repos: dict[str, list[str]] = defaultdict(list)
        for node, data in self.graph.nodes(data=True):
            if data.get("node_type") != "repository":
                continue
            for topic in data.get("topics", []):
                topic_repos[topic].append(node)

        added = 0
        seen = set()
        for repos in topic_repos.values():
            if len(repos) < 2:
                continue
            for i, r1 in enumerate(repos):
                for r2 in repos[i + 1 :]:
                    pair = tuple(sorted([r1, r2]))
                    if pair in seen:
                        continue

                    t1 = set(self.graph.nodes[r1].get("topics", []))
                    t2 = set(self.graph.nodes[r2].get("topics", []))
                    shared = len(t1 & t2)
                    if shared >= min_shared_topics:
                        seen.add(pair)
                        self.graph.add_edge(
                            r1, r2, edge_type="same_topic", weight=shared
                        )
                        self.graph.add_edge(
                            r2, r1, edge_type="same_topic", weight=shared
                        )
                        added += 1
        logger.info("Added %d topic-overlap edges", added)

    def add_event_nodes(self, events_df: pd.DataFrame):
        """Add event nodes and entity-event edges."""
        added = 0
        for _, row in events_df.iterrows():
            event_id = f"event_{row.get('event_type', '')}_{row.get('event_date', '')}_{added}"
            self.graph.add_node(
                event_id,
                node_type="event",
                event_type=row.get("event_type", ""),
                event_date=str(row.get("event_date", "")),
                title=row.get("title", ""),
                source=row.get("source", ""),
            )
            repo = row.get("repo_name", "")
            if repo in self.graph:
                self.graph.add_edge(repo, event_id, edge_type="mentioned_in")
            added += 1
        logger.info("Added %d event nodes", added)

    def get_neighbors(
        self,
        repo_name: str,
        edge_types: list[str] | None = None,
        max_neighbors: int = 20,
    ) -> list[str]:
        """Get neighbor repo names, optionally filtered by edge type."""
        if repo_name not in self.graph:
            return []

        neighbors = []
        for _, target, data in self.graph.edges(repo_name, data=True):
            if self.graph.nodes[target].get("node_type") != "repository":
                continue
            if edge_types and data.get("edge_type") not in edge_types:
                continue
            neighbors.append(target)
            if len(neighbors) >= max_neighbors:
                break

        return neighbors

    def summary(self) -> dict:
        """Return graph statistics."""
        node_types = defaultdict(int)
        for _, data in self.graph.nodes(data=True):
            node_types[data.get("node_type", "unknown")] += 1

        edge_types = defaultdict(int)
        for _, _, data in self.graph.edges(data=True):
            edge_types[data.get("edge_type", "unknown")] += 1

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": dict(node_types),
            "edge_types": dict(edge_types),
        }
