"""Libraries.io data loading for dependency graph construction.

Expects the Libraries.io Open Data snapshot to be downloaded and extracted
to data/raw/libraries-io/. Download from: https://libraries.io/data
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from pulsegraph.config import RAW_DIR

logger = logging.getLogger(__name__)

LIBRARIES_DIR = RAW_DIR / "libraries-io"


def load_dependencies(platform: str = "NPM") -> pd.DataFrame:
    """Load the dependencies CSV for a given platform.

    The Libraries.io dataset has a file called
    `dependencies-1.6.0-2024-XX-XX.csv` (version varies).
    """
    dep_files = list(LIBRARIES_DIR.glob("dependencies*.csv"))
    if not dep_files:
        logger.warning(
            "No dependencies CSV found in %s. Download from https://libraries.io/data",
            LIBRARIES_DIR,
        )
        return pd.DataFrame()

    dep_file = sorted(dep_files)[-1]
    logger.info("Loading dependencies from %s (filtering platform=%s)", dep_file, platform)

    cols = [
        "Platform", "Project Name", "Project ID",
        "Dependency Name", "Dependency Platform", "Dependency Kind",
    ]
    chunks = pd.read_csv(dep_file, usecols=cols, chunksize=500_000)

    filtered_chunks = []
    for chunk in chunks:
        mask = chunk["Platform"].str.upper() == platform.upper()
        filtered_chunks.append(chunk[mask])

    if not filtered_chunks:
        return pd.DataFrame()

    df = pd.concat(filtered_chunks, ignore_index=True)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    return df


def load_projects(platform: str = "NPM") -> pd.DataFrame:
    """Load the projects CSV for a given platform."""
    proj_files = list(LIBRARIES_DIR.glob("projects*.csv"))
    if not proj_files:
        logger.warning("No projects CSV found in %s", LIBRARIES_DIR)
        return pd.DataFrame()

    proj_file = sorted(proj_files)[-1]
    logger.info("Loading projects from %s (filtering platform=%s)", proj_file, platform)

    cols = [
        "Platform", "Name", "Repository URL", "Homepage URL",
        "Language", "Keywords", "Stars", "Forks",
    ]
    try:
        chunks = pd.read_csv(proj_file, usecols=cols, chunksize=500_000)
        filtered = []
        for chunk in chunks:
            mask = chunk["Platform"].str.upper() == platform.upper()
            filtered.append(chunk[mask])
        df = pd.concat(filtered, ignore_index=True) if filtered else pd.DataFrame()
    except Exception as e:
        logger.error("Failed to load projects: %s", e)
        return pd.DataFrame()

    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    return df


def build_github_dependency_edges(
    target_repos: list[str],
    platform: str = "NPM",
) -> pd.DataFrame:
    """Build dependency edges between GitHub repos in the target set.

    Returns a DataFrame with columns: source_repo, target_repo, edge_type.
    Only includes edges where both source and target are in target_repos.
    """
    projects = load_projects(platform)
    if projects.empty:
        return pd.DataFrame(columns=["source_repo", "target_repo", "edge_type"])

    projects["github_repo"] = (
        projects["repository_url"]
        .fillna("")
        .str.extract(r"github\.com/([^/]+/[^/]+?)(?:\.git)?$", expand=False)
        .str.lower()
    )
    projects = projects.dropna(subset=["github_repo"])

    name_to_repo = dict(zip(projects["name"], projects["github_repo"]))
    target_set = {r.lower() for r in target_repos}

    deps = load_dependencies(platform)
    if deps.empty:
        return pd.DataFrame(columns=["source_repo", "target_repo", "edge_type"])

    deps["source_repo"] = deps["project_name"].map(name_to_repo)
    deps["target_repo"] = deps["dependency_name"].map(name_to_repo)
    deps = deps.dropna(subset=["source_repo", "target_repo"])

    mask = deps["source_repo"].isin(target_set) & deps["target_repo"].isin(target_set)
    edges = deps[mask][["source_repo", "target_repo"]].copy()
    edges["edge_type"] = "depends_on"
    edges = edges.drop_duplicates()

    logger.info("Built %d dependency edges between target repos", len(edges))
    return edges
