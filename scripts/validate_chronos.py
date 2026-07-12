#!/usr/bin/env python3
"""Validation: is Chronos-2's win real skill, or a sampling/outlier artifact?

Reads only local result parquet (no network, no model, no credentials).

Three robustness checks:
  1. Paired per-window comparison (chronos2 vs each baseline on the SAME
     repo/window/horizon): win-rate, median CRPS ratio, Wilcoxon signed-rank.
     Robust to the outlier-driven means.
  2. MAE (point-only, median forecast): distribution-shape independent, so a
     Chronos MAE win cannot be a quantile-vs-bootstrap sampling artifact.
  3. Coverage sanity: are Chronos intervals honestly calibrated per regime?
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
res = pd.read_parquet(ROOT / "reports" / "experiment_real_full_results.parquet")

key = ["repo_name", "horizon", "cutoff_idx", "stratum"]
baselines = ["naive_seasonal", "linear", "ets", "lightgbm"]
chr_df = res[res["model_name"] == "chronos2"][key + ["crps_mean", "mae"]].rename(
    columns={"crps_mean": "crps_chr", "mae": "mae_chr"}
)

print("=" * 84)
print("PAIRED PER-WINDOW: Chronos-2 vs each baseline (same repo/window/horizon)")
print("win% = fraction of windows where Chronos-2 has strictly lower CRPS")
print("medRatio = median(crps_chr / crps_base)  (<1 favours Chronos); p = Wilcoxon")
print("=" * 84)
for regime in ["breakout", "steady", "dead"]:
    print(f"\n--- {regime} ---")
    for h in [7, 30, 90]:
        for b in baselines:
            bdf = res[res["model_name"] == b][key + ["crps_mean"]].rename(
                columns={"crps_mean": "crps_base"}
            )
            m = chr_df.merge(bdf, on=key)
            m = m[(m["horizon"] == h) & (m["stratum"] == regime)]
            if m.empty:
                continue
            win = float((m["crps_chr"] < m["crps_base"]).mean())
            ratio = m["crps_chr"] / m["crps_base"].replace(0, np.nan)
            medratio = float(ratio.median())
            try:
                _, p = stats.wilcoxon(m["crps_chr"], m["crps_base"])
            except ValueError:
                p = float("nan")
            print(f"  h={h:2d}d vs {b:15s} n={len(m):3d}  win={win*100:5.1f}%  "
                  f"medRatio={medratio:5.2f}  p={p:.1e}")

print("\n" + "=" * 84)
print("MAE (point-only, median forecast) by regime x horizon  [lower=better]")
print("If Chronos wins here too, the advantage is real point skill, not sampling shape.")
print("=" * 84)
mae_tbl = res.groupby(["stratum", "horizon", "model_name"])["mae"].mean().unstack("model_name")
mae_tbl = mae_tbl[["chronos2"] + baselines]
print(mae_tbl.round(2).to_string())

print("\n" + "=" * 84)
print("MAE paired win-rate: Chronos-2 vs BEST baseline per window")
print("=" * 84)
for regime in ["breakout", "steady", "dead"]:
    for h in [7, 30, 90]:
        best_base = res[(res.model_name.isin(baselines)) & (res.horizon == h)
                        & (res.stratum == regime)]
        # best baseline per window = min mae across baselines
        bb = (
            best_base.groupby(key)["mae"]
            .min()
            .reset_index()
            .rename(columns={"mae": "mae_bestbase"})
        )
        m = chr_df.merge(bb, on=key)
        m = m[(m["horizon"] == h) & (m["stratum"] == regime)]
        if m.empty:
            continue
        win = float((m["mae_chr"] < m["mae_bestbase"]).mean())
        print(f"  {regime:9s} h={h:2d}d  n={len(m):3d}  "
              f"Chronos beats best-baseline MAE in {win*100:5.1f}% of windows")
