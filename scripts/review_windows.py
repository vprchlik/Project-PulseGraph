#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Review audit 2: where do the backtest windows actually sit?

Because cutoffs run from day 180 to day 390 (8 windows, step 30) counting from
each repo's FIRST STAR, the evaluated period is always the series' early life.
This script quantifies:
  1. Calendar-date distribution of eval windows by regime (pretraining overlap).
  2. Position of the series' global peak relative to the eval span
     (is "breakout" evaluated on the spike, the decay, or neither?).
  3. Fraction of eval days that are zero, by regime (zero-inflation context).
  4. Scale distribution of eval-window actuals by regime.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pulsegraph.config import RAW_DIR, REPORTS_DIR
from pulsegraph.data.loader import build_dataset_for_forecasting, load_daily_signals

meta = pd.read_parquet(RAW_DIR / "ingest_metadata.parquet")
meta = meta[meta["status"] == "ok"]
strata = dict(zip(meta["repo_name"], meta["regime_label"]))

signals = load_daily_signals(RAW_DIR / "daily_signals.parquet")
dataset = build_dataset_for_forecasting(signals, list(strata.keys()),
                                        signal_col="stars", min_history_days=270)

res = pd.read_parquet(REPORTS_DIR / "experiment_real_full6_results.parquet")
one = res[res.model_name == "naive_seasonal"]
cut_by_repo = one.groupby("repo_name")["cutoff_idx"].agg(["min", "max"])

rows = []
for name, series in dataset.items():
    if name not in cut_by_repo.index:
        continue
    cmin, cmax = cut_by_repo.loc[name]
    eval_start = int(cmin)               # first forecasted day index
    eval_end = int(cmax) + 90            # last forecasted day index (h=90)
    eval_end = min(eval_end, len(series))
    vals = series.values.astype(float)
    peak_idx = int(np.argmax(vals))
    eval_vals = vals[eval_start:eval_end]
    rows.append({
        "repo": name,
        "regime": strata[name],
        "series_len": len(series),
        "first_date": series.index[0],
        "eval_start_date": series.index[eval_start],
        "eval_end_date": series.index[eval_end - 1],
        "peak_idx": peak_idx,
        "peak_in_context_only": peak_idx < eval_start,
        "peak_in_eval": eval_start <= peak_idx < eval_end,
        "peak_after_eval": peak_idx >= eval_end,
        "eval_frac_zero": float((eval_vals == 0).mean()),
        "eval_mean": float(eval_vals.mean()),
        "eval_max": float(eval_vals.max()),
        "series_mean": float(vals.mean()),
    })
df = pd.DataFrame(rows)

print("=" * 90)
print("EVAL WINDOW CALENDAR SPAN BY REGIME (eval = forecasted days, idx 180..~480)")
print("=" * 90)
for r in ["breakout", "steady", "dead"]:
    g = df[df.regime == r]
    print(f"\n{r} (n={len(g)}):")
    print(f"  eval start dates: min={g.eval_start_date.min().date()}  "
          f"median={g.eval_start_date.median().date()}  max={g.eval_start_date.max().date()}")
    print(f"  eval end dates:   min={g.eval_end_date.min().date()}  "
          f"median={g.eval_end_date.median().date()}  max={g.eval_end_date.max().date()}")
    for boundary in ["2024-01-01", "2025-01-01", "2025-11-01"]:
        frac = (g.eval_end_date >= pd.Timestamp(boundary)).mean()
        print(f"  repos with any eval day >= {boundary}: {frac*100:5.1f}%")

print("\n" + "=" * 90)
print("PEAK POSITION vs EVAL SPAN (is the spike in context, in eval, or later?)")
print("=" * 90)
for r in ["breakout", "steady", "dead"]:
    g = df[df.regime == r]
    print(f"{r:9s} peak in context-only: {g.peak_in_context_only.mean()*100:5.1f}%   "
          f"peak inside eval span: {g.peak_in_eval.mean()*100:5.1f}%   "
          f"peak after eval span: {g.peak_after_eval.mean()*100:5.1f}%")

print("\n" + "=" * 90)
print("EVAL-WINDOW SIGNAL CHARACTER BY REGIME")
print("=" * 90)
for r in ["breakout", "steady", "dead"]:
    g = df[df.regime == r]
    print(f"{r:9s} frac zero days in eval: median={g.eval_frac_zero.median()*100:5.1f}%  "
          f"eval mean stars/day: median={g.eval_mean.median():7.2f}  "
          f"eval max stars/day: median={g.eval_max.median():7.1f}")

# Calendar overlap with model training cutoffs (approx public release dates):
print("\n" + "=" * 90)
print("WINDOW-LEVEL CALENDAR DATES (all 1374 windows, from cutoff_date)")
print("=" * 90)
one = one.copy()
one["cutoff_dt"] = pd.to_datetime(one["cutoff_date"])
for r in ["breakout", "steady", "dead"]:
    g = one[one.stratum == r].drop_duplicates(subset=["repo_name", "cutoff_idx"])
    q = g["cutoff_dt"].quantile([0.1, 0.5, 0.9]).dt.date.tolist()
    frac24 = (g["cutoff_dt"] >= "2024-01-01").mean()
    frac25 = (g["cutoff_dt"] >= "2025-06-01").mean()
    print(f"{r:9s} n={len(g):4d}  cutoff dates p10/p50/p90: {q}  "
          f">=2024: {frac24*100:4.1f}%  >=2025-06: {frac25*100:4.1f}%")
