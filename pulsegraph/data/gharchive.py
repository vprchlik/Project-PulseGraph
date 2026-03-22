"""GH Archive data access via BigQuery.

Reconstructs daily event counts (stars, forks, issues, PRs, commits) for
target repositories from the public `githubarchive` dataset on BigQuery.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

from pulsegraph.config import settings, RAW_DIR

logger = logging.getLogger(__name__)

DAILY_STARS_QUERY = """
SELECT
    repo.name AS repo_name,
    DATE(created_at) AS event_date,
    COUNT(*) AS star_count
FROM `githubarchive.day.{table_suffix}`
WHERE type = 'WatchEvent'
GROUP BY repo_name, event_date
"""

TOP_REPOS_QUERY = """
SELECT
    repo.name AS repo_name,
    COUNT(*) AS total_stars
FROM `githubarchive.day.2*`
WHERE type = 'WatchEvent'
    AND _TABLE_SUFFIX BETWEEN '{start}' AND '{end}'
GROUP BY repo_name
ORDER BY total_stars DESC
LIMIT {limit}
"""

TAIL_REPOS_QUERY = """
SELECT
    repo.name AS repo_name,
    COUNT(*) AS total_stars
FROM `githubarchive.day.2*`
WHERE type = 'WatchEvent'
    AND _TABLE_SUFFIX BETWEEN '{start}' AND '{end}'
GROUP BY repo_name
HAVING total_stars BETWEEN {star_min} AND {star_max}
ORDER BY RAND()
LIMIT {limit}
"""

DAILY_SIGNALS_QUERY = """
SELECT
    repo.name AS repo_name,
    DATE(created_at) AS event_date,
    type AS event_type,
    COUNT(*) AS event_count
FROM `githubarchive.day.2*`
WHERE _TABLE_SUFFIX BETWEEN '{start}' AND '{end}'
    AND repo.name IN UNNEST(@repo_names)
    AND type IN (
        'WatchEvent',       -- stars
        'ForkEvent',        -- forks
        'PushEvent',        -- commits (approximation)
        'IssuesEvent',      -- issues opened/closed
        'PullRequestEvent', -- PRs
        'ReleaseEvent'      -- releases
    )
GROUP BY repo_name, event_date, event_type
ORDER BY repo_name, event_date
"""


class GHArchiveClient:
    """Client for querying GH Archive data from BigQuery."""

    def __init__(self, project_id: str | None = None):
        self.project_id = project_id or settings.bigquery_project_id
        self._client: bigquery.Client | None = None

    @property
    def client(self) -> bigquery.Client:
        if self._client is None:
            self._client = bigquery.Client(project=self.project_id)
        return self._client

    def get_top_repos(
        self,
        limit: int = 10_000,
        start_date: date = date(2022, 1, 1),
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """Retrieve the top N repos by total star events in the date range."""
        if end_date is None:
            end_date = date.today()
        start_suffix = start_date.strftime("%Y%m%d")
        end_suffix = end_date.strftime("%Y%m%d")

        query = TOP_REPOS_QUERY.format(
            start=start_suffix, end=end_suffix, limit=limit
        )
        logger.info("Querying top %d repos from GH Archive (%s to %s)", limit, start_date, end_date)
        df = self.client.query(query).to_dataframe()
        return df

    def get_tail_repos(
        self,
        limit: int = 5_000,
        star_min: int = 100,
        star_max: int = 1_000,
        start_date: date = date(2022, 1, 1),
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """Retrieve a random sample of repos in the tail star range."""
        if end_date is None:
            end_date = date.today()
        start_suffix = start_date.strftime("%Y%m%d")
        end_suffix = end_date.strftime("%Y%m%d")

        query = TAIL_REPOS_QUERY.format(
            start=start_suffix,
            end=end_suffix,
            limit=limit,
            star_min=star_min,
            star_max=star_max,
        )
        logger.info("Querying tail repos (%d-%d stars) from GH Archive", star_min, star_max)
        df = self.client.query(query).to_dataframe()
        return df

    def get_daily_signals(
        self,
        repo_names: list[str],
        start_date: date = date(2022, 1, 1),
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """Fetch daily event counts for the given repos, pivoted by event type."""
        if end_date is None:
            end_date = date.today()
        start_suffix = start_date.strftime("%Y%m%d")
        end_suffix = end_date.strftime("%Y%m%d")

        query = DAILY_SIGNALS_QUERY.format(start=start_suffix, end=end_suffix)
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("repo_names", "STRING", repo_names),
            ]
        )
        logger.info("Fetching daily signals for %d repos", len(repo_names))
        df = self.client.query(query, job_config=job_config).to_dataframe()

        pivot = df.pivot_table(
            index=["repo_name", "event_date"],
            columns="event_type",
            values="event_count",
            fill_value=0,
        ).reset_index()

        pivot.columns = [str(c) for c in pivot.columns]
        rename_map = {
            "WatchEvent": "stars",
            "ForkEvent": "forks",
            "PushEvent": "pushes",
            "IssuesEvent": "issues",
            "PullRequestEvent": "pull_requests",
            "ReleaseEvent": "releases",
        }
        pivot.rename(columns=rename_map, inplace=True)
        return pivot

    def backfill_to_parquet(
        self,
        repo_names: list[str],
        start_date: date = date(2022, 1, 1),
        end_date: date | None = None,
        output_path: Path | None = None,
    ) -> Path:
        """Pull daily signals and save as a parquet file."""
        df = self.get_daily_signals(repo_names, start_date, end_date)
        if output_path is None:
            output_path = RAW_DIR / "daily_signals.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info("Saved %d rows to %s", len(df), output_path)
        return output_path


def build_target_repo_list(
    top_n: int = 10_000,
    tail_n: int = 5_000,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Build and save the combined target repo list (top + tail sample)."""
    client = GHArchiveClient()

    top_df = client.get_top_repos(limit=top_n)
    top_df["stratum"] = "high"

    tail_df = client.get_tail_repos(limit=tail_n)
    tail_df["stratum"] = "tail"

    combined = pd.concat([top_df, tail_df], ignore_index=True)
    combined = combined.drop_duplicates(subset="repo_name", keep="first")

    if output_path is None:
        output_path = RAW_DIR / "target_repos.parquet"
    combined.to_parquet(output_path, index=False)
    logger.info("Built target repo list: %d repos", len(combined))
    return combined
