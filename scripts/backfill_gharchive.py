#!/usr/bin/env python3
"""Backfill historical data from GH Archive via BigQuery.

Requires:
- GOOGLE_APPLICATION_CREDENTIALS env var pointing to a service account JSON
- BIGQUERY_PROJECT_ID env var

Usage:
    python scripts/backfill_gharchive.py --top-n 10000 --tail-n 5000
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pulsegraph.config import RAW_DIR
from pulsegraph.data.gharchive import GHArchiveClient, build_target_repo_list

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backfill")


def main():
    parser = argparse.ArgumentParser(description="Backfill GH Archive data")
    parser.add_argument("--top-n", type=int, default=10_000)
    parser.add_argument("--tail-n", type=int, default=5_000)
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()

    start = date.fromisoformat(args.start_date)

    logger.info("Step 1: Building target repo list")
    repos_df = build_target_repo_list(top_n=args.top_n, tail_n=args.tail_n)
    repo_names = repos_df["repo_name"].tolist()
    logger.info("Target list: %d repos", len(repo_names))

    logger.info("Step 2: Backfilling daily signals in batches of %d", args.batch_size)
    client = GHArchiveClient()

    all_dfs = []
    for i in range(0, len(repo_names), args.batch_size):
        batch = repo_names[i : i + args.batch_size]
        logger.info("  Batch %d-%d / %d", i, i + len(batch), len(repo_names))
        df = client.get_daily_signals(batch, start_date=start)
        all_dfs.append(df)

    import pandas as pd

    combined = pd.concat(all_dfs, ignore_index=True)
    output_path = RAW_DIR / "daily_signals.parquet"
    combined.to_parquet(output_path, index=False)
    logger.info("Saved %d rows to %s", len(combined), output_path)


if __name__ == "__main__":
    main()
