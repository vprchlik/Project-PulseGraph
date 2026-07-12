#!/usr/bin/env python3
"""Quantify observation-window censoring in GitHub star series.

The loader currently builds each series from its first observed star through its
last observed star. This script compares those event endpoints with repository
creation time and the ingestion snapshot date to measure omitted leading and
trailing zero runs. It is read-only.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

signals = pd.read_parquet(RAW / "daily_signals.parquet")
meta = pd.read_parquet(RAW / "ingest_metadata.parquet")
meta = meta[meta["status"] == "ok"].copy()

signals["event_date"] = pd.to_datetime(signals["event_date"]).dt.normalize()
endpoints = signals.groupby("repo_name")["event_date"].agg(
    first_star="min", last_star="max"
)
audit = meta.set_index("repo_name").join(endpoints, how="inner")
audit["created_date"] = (
    pd.to_datetime(audit["created_at"], utc=True)
    .dt.tz_localize(None)
    .dt.normalize()
)

# The latest date in this ingestion snapshot is the conservative common
# observation endpoint. Using it may overstate a trailing gap by at most one day
# for requests that crossed UTC midnight.
snapshot_date = signals["event_date"].max()
audit["leading_gap_days"] = (audit["first_star"] - audit["created_date"]).dt.days
audit["trailing_gap_days"] = (snapshot_date - audit["last_star"]).dt.days

print(f"Conservative snapshot endpoint: {snapshot_date.date()}")
print("\nEndpoint gaps by discovery-time label:")
summary = audit.groupby("regime_label").agg(
    n=("leading_gap_days", "size"),
    leading_median=("leading_gap_days", "median"),
    leading_p95=("leading_gap_days", lambda x: x.quantile(0.95)),
    trailing_median=("trailing_gap_days", "median"),
    trailing_p95=("trailing_gap_days", lambda x: x.quantile(0.95)),
    trailing_max=("trailing_gap_days", "max"),
)
print(summary.round(1).to_string())

for threshold in [7, 30, 90]:
    frac = (
        audit.groupby("regime_label")["trailing_gap_days"]
        .apply(lambda x: float((x >= threshold).mean()))
        .rename(f"fraction_trailing_at_least_{threshold}d")
    )
    print(f"\n{frac.name}:")
    print(frac.round(3).to_string())

print("\nLargest trailing gaps:")
print(
    audit.sort_values("trailing_gap_days", ascending=False)[
        ["regime_label", "created_date", "first_star", "last_star", "trailing_gap_days"]
    ].head(12).to_string()
)

out = ROOT / "reports" / "review_censoring.csv"
audit.reset_index().to_csv(out, index=False)
print(f"\nSaved {out}")
