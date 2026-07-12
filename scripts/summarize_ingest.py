#!/usr/bin/env python3
"""Summarize the ingested real GitHub star data for the verification checkpoint.

Reads only local parquet files (no network, no credentials).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

daily = pd.read_parquet(RAW / "daily_signals.parquet")
meta = pd.read_parquet(RAW / "ingest_metadata.parquet")
daily["event_date"] = pd.to_datetime(daily["event_date"])

ok = meta[meta["status"] == "ok"].copy()
print("TOTAL repos OK:", len(ok), "| daily rows:", len(daily))
print("global date range:", daily["event_date"].min().date(), "->",
      daily["event_date"].max().date())
print()

print("=== per-regime: count, median total stars, median span, usable(>=270d) ===")
for r in ["breakout", "steady", "dead"]:
    sub = ok[ok["regime_label"] == r]
    usable = int((sub["n_days_span"] >= 270).sum())
    print(f"{r:9s} n={len(sub):3d}  med_stars={int(sub['n_stars_pulled'].median()):6d}  "
          f"med_span={int(sub['n_days_span'].median()):5d}d  usable(>=270d)={usable:3d}")
print()

print("=== SPOT-CHECK: one repo per regime (median-sized, >=270d history) ===")
for r in ["breakout", "steady", "dead"]:
    sub = ok[(ok["regime_label"] == r) & (ok["n_days_span"] >= 270)].sort_values(
        "n_stars_pulled", ascending=False
    )
    if sub.empty:
        sub = ok[ok["regime_label"] == r].sort_values("n_stars_pulled", ascending=False)
    row = sub.iloc[len(sub) // 2]
    name = row["repo_name"]
    s = daily[daily["repo_name"] == name].sort_values("event_date")
    first5 = list(zip([d.date().isoformat() for d in s["event_date"].head()],
                      s["stars"].head().astype(int)))
    last5 = list(zip([d.date().isoformat() for d in s["event_date"].tail()],
                     s["stars"].tail().astype(int)))
    print(f"[{r}] {name}")
    print(f"    total stars pulled: {int(row['n_stars_pulled'])}  | "
          f"current_stars(meta): {int(row['current_stars'])}")
    print(f"    date range: {s['event_date'].min().date()} -> {s['event_date'].max().date()}  "
          f"({int(row['n_days_span'])} days span, {len(s)} active days)")
    print(f"    first 5 daily: {first5}")
    print(f"    last 5 daily:  {last5}")
    print(f"    peak day: {s.loc[s['stars'].idxmax(), 'event_date'].date()} = "
          f"{int(s['stars'].max())} stars; mean={s['stars'].mean():.1f}/day")
    print()
