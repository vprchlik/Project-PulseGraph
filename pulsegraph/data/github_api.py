"""GitHub REST/GraphQL API client for incremental data updates."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import httpx
import pandas as pd

from pulsegraph.config import settings

logger = logging.getLogger(__name__)

BASE_URL = "https://api.github.com"
GRAPHQL_URL = "https://api.github.com/graphql"


class GitHubAPIClient:
    """Thin wrapper around the GitHub REST and GraphQL APIs."""

    def __init__(self, token: str | None = None):
        self.token = token or settings.github_token
        self._http: httpx.Client | None = None

    @property
    def http(self) -> httpx.Client:
        if self._http is None or self._http.is_closed:
            headers = {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            self._http = httpx.Client(
                base_url=BASE_URL, headers=headers, timeout=30.0
            )
        return self._http

    def close(self):
        if self._http and not self._http.is_closed:
            self._http.close()

    def _get_paginated(self, url: str, params: dict | None = None, max_pages: int = 10) -> list:
        """Follow pagination links up to max_pages."""
        results = []
        params = params or {}
        params.setdefault("per_page", 100)

        for _ in range(max_pages):
            resp = self.http.get(url, params=params)
            self._handle_rate_limit(resp)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break
            results.extend(data)
            link = resp.headers.get("Link", "")
            if 'rel="next"' not in link:
                break
            next_url = [
                part.split(";")[0].strip("<> ")
                for part in link.split(",")
                if 'rel="next"' in part
            ][0]
            url = next_url
            params = {}
        return results

    def _handle_rate_limit(self, resp: httpx.Response):
        remaining = int(resp.headers.get("X-RateLimit-Remaining", 999))
        if remaining < 10:
            reset_ts = int(resp.headers.get("X-RateLimit-Reset", 0))
            sleep_for = max(0, reset_ts - time.time()) + 1
            logger.warning("Rate limit near exhaustion, sleeping %.0fs", sleep_for)
            time.sleep(sleep_for)

    def get_repo_info(self, owner: str, repo: str) -> dict:
        """Fetch repo metadata."""
        resp = self.http.get(f"/repos/{owner}/{repo}")
        self._handle_rate_limit(resp)
        resp.raise_for_status()
        return resp.json()

    def get_repo_releases(self, owner: str, repo: str, max_pages: int = 5) -> list[dict]:
        return self._get_paginated(f"/repos/{owner}/{repo}/releases", max_pages=max_pages)

    def get_repo_contributors(self, owner: str, repo: str, max_pages: int = 3) -> list[dict]:
        return self._get_paginated(f"/repos/{owner}/{repo}/contributors", max_pages=max_pages)

    def get_repo_topics(self, owner: str, repo: str) -> list[str]:
        resp = self.http.get(f"/repos/{owner}/{repo}/topics")
        self._handle_rate_limit(resp)
        resp.raise_for_status()
        return resp.json().get("names", [])

    def get_repo_metadata_batch(self, repo_full_names: list[str]) -> pd.DataFrame:
        """Fetch metadata for a batch of repos (owner/name format)."""
        records = []
        for full_name in repo_full_names:
            parts = full_name.split("/")
            if len(parts) != 2:
                logger.warning("Skipping invalid repo name: %s", full_name)
                continue
            owner, repo = parts
            try:
                info = self.get_repo_info(owner, repo)
                records.append({
                    "repo_name": full_name,
                    "description": info.get("description", ""),
                    "language": info.get("language", ""),
                    "license": (info.get("license") or {}).get("spdx_id", ""),
                    "created_at": info.get("created_at", ""),
                    "stargazers_count": info.get("stargazers_count", 0),
                    "forks_count": info.get("forks_count", 0),
                    "open_issues_count": info.get("open_issues_count", 0),
                    "topics": info.get("topics", []),
                    "org": owner,
                    "default_branch": info.get("default_branch", "main"),
                    "archived": info.get("archived", False),
                    "fork": info.get("fork", False),
                })
            except httpx.HTTPStatusError as e:
                logger.warning("Failed to fetch %s: %s", full_name, e)
        return pd.DataFrame(records)

    def graphql_query(self, query: str, variables: dict | None = None) -> dict:
        """Execute a raw GraphQL query."""
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        resp = httpx.post(
            GRAPHQL_URL,
            json={"query": query, "variables": variables or {}},
            headers=headers,
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        if "errors" in data:
            logger.error("GraphQL errors: %s", data["errors"])
        return data.get("data", {})
