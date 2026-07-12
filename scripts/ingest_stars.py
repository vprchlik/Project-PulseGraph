#!/usr/bin/env python3
"""Ingest real daily GitHub star history via the REST stargazers API.

Reconstructs the true daily count of new stars per repo from per-star
``starred_at`` timestamps, deliberately sampling across three regimes so the
shock-driven (spiky) cases the study targets are well represented:

  * breakout  - recently created repos that already have high star counts
                (high star velocity -> sharp viral spikes)
  * steady    - long-lived repos with moderate stars, still actively pushed
  * dead       - repos not pushed in years and/or archived (flat / decayed)

Repos are discovered reproducibly via the Search API (stratified queries),
not hand-picked, to avoid cherry-picking. Output matches the schema the
forecasting loader expects: columns [repo_name, event_date, stars].

Important constraint: the REST stargazers endpoint caps pagination at 400
pages (40,000 stargazers). Repos above that are left-truncated; we record a
``truncated`` flag per repo and keep star ceilings below 40k by default so
most series are complete.

Usage:
    python scripts/ingest_stars.py --per-regime 80
    python scripts/ingest_stars.py --per-regime 100 --max-stars 30000
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pulsegraph.config import RAW_DIR
from pulsegraph.data.github_api import GitHubAPIClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("ingest_stars")


def build_regime_queries(min_stars: int, max_stars: int) -> dict[str, str]:
    """Stratified Search API queries, one per regime.

    Dates are relative to "now" so the script stays reproducible over time.
    """
    today = datetime.now(timezone.utc).date()
    year = today.year

    return {
        # Created recently but already highly starred -> breakout velocity.
        "breakout": (
            f"stars:{max(min_stars, 3000)}..{max_stars} "
            f"created:>{year - 2}-01-01 sort:stars"
        ),
        # Old, moderate stars, still maintained -> steady growth.
        "steady": (
            f"stars:{min_stars}..{max_stars} "
            f"created:<{year - 6}-01-01 pushed:>{year - 1}-06-01"
        ),
        # Not pushed in years -> dead / flat.
        "dead": (
            f"stars:{min_stars}..{min(max_stars, 12000)} "
            f"pushed:<{year - 4}-01-01"
        ),
    }


def discover_repos(
    client: GitHubAPIClient,
    per_regime: int,
    min_stars: int,
    max_stars: int,
) -> pd.DataFrame:
    """Discover repos per regime via stratified search."""
    queries = build_regime_queries(min_stars, max_stars)
    rows: list[dict] = []
    seen: set[str] = set()

    for regime, query in queries.items():
        logger.info("Searching regime=%s query=%r", regime, query)
        items = client.search_repositories(query, max_results=per_regime * 2)
        kept = 0
        for it in items:
            full = it["full_name"]
            if full in seen:
                continue
            if it.get("fork"):
                continue
            seen.add(full)
            rows.append(
                {
                    "repo_name": full,
                    "regime_label": regime,
                    "current_stars": it.get("stargazers_count", 0),
                    "created_at": it.get("created_at", ""),
                    "pushed_at": it.get("pushed_at", ""),
                    "archived": it.get("archived", False),
                    "language": it.get("language") or "",
                }
            )
            kept += 1
            if kept >= per_regime:
                break
        logger.info("  regime=%s: kept %d repos", regime, kept)

    return pd.DataFrame(rows)


def ingest_daily_stars(
    client: GitHubAPIClient, repos: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch stargazer timestamps and build daily star-delta series."""
    daily_rows: list[dict] = []
    meta_rows: list[dict] = []

    for i, row in enumerate(repos.itertuples(index=False), start=1):
        full = row.repo_name
        owner, repo = full.split("/", 1)
        logger.info("[%d/%d] %s (%s)", i, len(repos), full, row.regime_label)
        try:
            timestamps, truncated = client.get_stargazer_timestamps(owner, repo)
        except Exception as e:  # noqa: BLE001
            logger.warning("  failed: %s", e)
            meta_rows.append({**row._asdict(), "n_stars_pulled": 0,
                              "truncated": False, "status": f"error: {e}"})
            continue

        if not timestamps:
            logger.warning("  no timestamps returned")
            meta_rows.append({**row._asdict(), "n_stars_pulled": 0,
                              "truncated": truncated, "status": "empty"})
            continue

        ts = pd.to_datetime(pd.Series(timestamps), utc=True, errors="coerce").dropna()
        daily = ts.dt.floor("D").value_counts().sort_index()
        for date, count in daily.items():
            daily_rows.append(
                {
                    "repo_name": full,
                    "event_date": date.tz_localize(None),
                    "stars": int(count),
                }
            )
        meta_rows.append(
            {
                **row._asdict(),
                "n_stars_pulled": int(len(ts)),
                "truncated": truncated,
                "first_date": daily.index.min().tz_localize(None),
                "last_date": daily.index.max().tz_localize(None),
                "n_days_span": int((daily.index.max() - daily.index.min()).days) + 1,
                "status": "ok",
            }
        )
        logger.info(
            "  pulled %d stars over %s -> %s%s",
            len(ts), daily.index.min().date(), daily.index.max().date(),
            " (TRUNCATED)" if truncated else "",
        )

    return pd.DataFrame(daily_rows), pd.DataFrame(meta_rows)


def main():
    parser = argparse.ArgumentParser(description="Ingest real daily GitHub star history")
    parser.add_argument("--per-regime", type=int, default=80,
                        help="Repos to keep per regime (breakout/steady/dead)")
    parser.add_argument("--min-stars", type=int, default=500)
    parser.add_argument("--max-stars", type=int, default=35000)
    parser.add_argument("--out", default=str(RAW_DIR / "daily_signals.parquet"))
    args = parser.parse_args()

    client = GitHubAPIClient()
    if not client.token:
        logger.error(
            "No GITHUB_TOKEN configured. Set it in .env or the environment. "
            "Unauthenticated access is rate-limited to 60 req/hr and cannot "
            "complete this ingestion."
        )
        sys.exit(1)

    repos = discover_repos(client, args.per_regime, args.min_stars, args.max_stars)
    if repos.empty:
        logger.error("Discovery returned no repos. Aborting.")
        sys.exit(1)

    repos_path = RAW_DIR / "target_repos.parquet"
    repos.to_parquet(repos_path, index=False)
    logger.info("Discovered %d repos -> %s", len(repos), repos_path)

    daily_df, meta_df = ingest_daily_stars(client, repos)
    client.close()

    if daily_df.empty:
        logger.error("No daily data ingested. Aborting.")
        sys.exit(1)

    out_path = Path(args.out)
    daily_df.to_parquet(out_path, index=False)
    meta_path = RAW_DIR / "ingest_metadata.parquet"
    meta_df.to_parquet(meta_path, index=False)

    ok = meta_df[meta_df["status"] == "ok"]
    logger.info("=" * 60)
    logger.info("Ingestion complete: %d/%d repos OK", len(ok), len(meta_df))
    logger.info("  daily rows: %d -> %s", len(daily_df), out_path)
    logger.info("  metadata:   %s", meta_path)
    if not ok.empty:
        logger.info("  truncated (>40k stars): %d", int(ok["truncated"].sum()))
        logger.info("  by regime: %s", ok["regime_label"].value_counts().to_dict())


if __name__ == "__main__":
    main()
