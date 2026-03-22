#!/usr/bin/env python3
"""Daily incremental ingestion from GitHub API.

Designed to be run as a daily cron job to keep time-series data current.
Appends new data points to the existing parquet file.

Usage:
    python scripts/daily_ingest.py
    
Crontab (run daily at 2 AM):
    0 2 * * * cd /path/to/PulseGraph && python scripts/daily_ingest.py
"""

from __future__ import annotations

import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pulsegraph.config import RAW_DIR
from pulsegraph.data.github_api import GitHubAPIClient
from pulsegraph.data.loader import load_target_repos

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("daily_ingest")


def main():
    repos_path = RAW_DIR / "target_repos.parquet"
    if not repos_path.exists():
        logger.error("No target repos file at %s. Run backfill first.", repos_path)
        return

    repos_df = load_target_repos(repos_path)
    repo_names = repos_df["repo_name"].tolist()
    logger.info("Loaded %d target repos", len(repo_names))

    client = GitHubAPIClient()
    today = date.today()

    signals_path = RAW_DIR / "daily_signals.parquet"
    if signals_path.exists():
        existing = pd.read_parquet(signals_path)
        logger.info("Existing data: %d rows", len(existing))
    else:
        existing = pd.DataFrame()

    logger.info("Fetching metadata for repos (this updates current star counts)")
    batch_size = 100
    new_records = []

    for i in range(0, len(repo_names), batch_size):
        batch = repo_names[i : i + batch_size]
        logger.info("  Batch %d-%d / %d", i, i + len(batch), len(repo_names))

        for full_name in batch:
            parts = full_name.split("/")
            if len(parts) != 2:
                continue
            try:
                info = client.get_repo_info(parts[0], parts[1])
                new_records.append({
                    "repo_name": full_name,
                    "event_date": today,
                    "stars": info.get("stargazers_count", 0),
                    "forks": info.get("forks_count", 0),
                    "pushes": 0,
                    "issues": info.get("open_issues_count", 0),
                    "pull_requests": 0,
                    "releases": 0,
                })
            except Exception as e:
                logger.warning("Failed to fetch %s: %s", full_name, e)

    if new_records:
        new_df = pd.DataFrame(new_records)
        new_df["event_date"] = pd.to_datetime(new_df["event_date"])

        if not existing.empty:
            existing["event_date"] = pd.to_datetime(existing["event_date"])
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["repo_name", "event_date"], keep="last"
            )
        else:
            combined = new_df

        combined.to_parquet(signals_path, index=False)
        logger.info("Saved %d total rows to %s", len(combined), signals_path)
    else:
        logger.warning("No new data fetched")

    client.close()


if __name__ == "__main__":
    main()
