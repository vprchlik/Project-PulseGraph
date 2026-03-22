"""Event ingestion from HN, Reddit, and GitHub releases."""

from __future__ import annotations

import logging
from datetime import date, datetime

import httpx
import pandas as pd

from pulsegraph.data.github_api import GitHubAPIClient

logger = logging.getLogger(__name__)

HN_SEARCH_URL = "https://hn.algolia.com/api/v1/search"


def fetch_hn_mentions(
    query: str,
    start_date: date | None = None,
    end_date: date | None = None,
    max_hits: int = 200,
) -> list[dict]:
    """Search Hacker News for posts mentioning a query (repo name or URL)."""
    params: dict = {
        "query": query,
        "tags": "story",
        "hitsPerPage": min(max_hits, 200),
    }
    if start_date:
        params["numericFilters"] = f"created_at_i>{int(datetime.combine(start_date, datetime.min.time()).timestamp())}"

    try:
        resp = httpx.get(HN_SEARCH_URL, params=params, timeout=15.0)
        resp.raise_for_status()
        hits = resp.json().get("hits", [])
    except (httpx.HTTPError, Exception) as e:
        logger.warning("HN search failed for %r: %s", query, e)
        return []

    results = []
    for hit in hits:
        created = hit.get("created_at", "")
        results.append({
            "event_type": "hn_mention",
            "event_date": created[:10] if created else "",
            "title": hit.get("title", ""),
            "url": f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}",
            "body": "",
            "source": "hackernews",
            "metadata": {
                "points": hit.get("points", 0),
                "num_comments": hit.get("num_comments", 0),
                "hn_id": hit.get("objectID", ""),
            },
        })
    return results


def fetch_github_releases(
    repo_full_name: str,
    gh_client: GitHubAPIClient | None = None,
) -> list[dict]:
    """Fetch release events for a GitHub repo."""
    client = gh_client or GitHubAPIClient()
    parts = repo_full_name.split("/")
    if len(parts) != 2:
        return []
    owner, repo = parts

    try:
        releases = client.get_repo_releases(owner, repo)
    except Exception as e:
        logger.warning("Failed to fetch releases for %s: %s", repo_full_name, e)
        return []

    results = []
    for rel in releases:
        pub = rel.get("published_at", "") or rel.get("created_at", "")
        results.append({
            "event_type": "release",
            "event_date": pub[:10] if pub else "",
            "title": rel.get("name", "") or rel.get("tag_name", ""),
            "url": rel.get("html_url", ""),
            "body": (rel.get("body", "") or "")[:2000],
            "source": "github",
            "metadata": {
                "tag": rel.get("tag_name", ""),
                "prerelease": rel.get("prerelease", False),
                "draft": rel.get("draft", False),
            },
        })
    return results


def fetch_events_for_repo(
    repo_full_name: str,
    gh_client: GitHubAPIClient | None = None,
    include_hn: bool = True,
) -> pd.DataFrame:
    """Aggregate all event sources for a single repo into a DataFrame."""
    events: list[dict] = []

    events.extend(fetch_github_releases(repo_full_name, gh_client))

    if include_hn:
        events.extend(fetch_hn_mentions(repo_full_name))
        repo_short = repo_full_name.split("/")[-1]
        if repo_short != repo_full_name:
            events.extend(fetch_hn_mentions(repo_short))

    if not events:
        return pd.DataFrame(columns=[
            "event_type", "event_date", "title", "url", "body", "source", "metadata"
        ])

    df = pd.DataFrame(events)
    df["repo_name"] = repo_full_name
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce").dt.date
    df = df.dropna(subset=["event_date"])
    df = df.drop_duplicates(subset=["event_type", "event_date", "title"])
    return df
