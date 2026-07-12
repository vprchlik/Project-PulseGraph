#!/usr/bin/env python3
"""Scale-robust audit of the final six-model backtest.

This is a read-only analysis over the stored per-window result artifact. It
checks whether the headline mean-CRPS result is driven by a few high-volume
repositories by reporting:

* macro averages after first averaging within each repository;
* per-repository skill relative to the stored naive-seasonal forecast;
* the fraction of repositories on which each model beats that baseline;
* leave-one-repository-out robustness of each regime/horizon winner; and
* concentration of each foundation model's gain over the best *fixed* baseline
  in that regime/horizon cell.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "reports" / "experiment_real_full6_results.parquet"

MODELS = [
    "naive_seasonal",
    "linear",
    "ets",
    "lightgbm",
    "chronos2",
    "timesfm",
]
BASELINES = MODELS[:4]
FOUNDATION = MODELS[-2:]

res = pd.read_parquet(RESULTS)

# Average windows within repository first so every repository has equal weight.
repo = (
    res.groupby(["stratum", "horizon", "repo_name", "model_name"], as_index=False)
    ["crps_mean"]
    .mean()
)
wide = repo.pivot(
    index=["stratum", "horizon", "repo_name"],
    columns="model_name",
    values="crps_mean",
)

print("=" * 96)
print("MACRO CRPS: average each repository first, then average repositories")
print("=" * 96)
macro = repo.groupby(["stratum", "horizon", "model_name"])["crps_mean"].mean().unstack()
print(macro[MODELS].round(3).to_string())

print("\n" + "=" * 96)
print("PER-REPOSITORY SKILL VS STORED NAIVE-SEASONAL: 1 - model/naive")
print("Positive is better; each repository receives equal weight.")
print("=" * 96)
records: list[dict] = []
for (stratum, horizon), group in wide.groupby(level=["stratum", "horizon"]):
    denom = group["naive_seasonal"].replace(0, np.nan)
    for model in MODELS[1:]:
        skill = 1.0 - group[model] / denom
        records.append(
            {
                "stratum": stratum,
                "horizon": int(horizon),
                "model": model,
                "mean_skill": float(skill.mean()),
                "median_skill": float(skill.median()),
                "repo_win_rate": float((group[model] < group["naive_seasonal"]).mean()),
                "n_repos": int(skill.notna().sum()),
            }
        )
skill_df = pd.DataFrame(records)
for stratum in ["breakout", "steady", "dead"]:
    print(f"\n--- {stratum} ---")
    print(
        skill_df[skill_df.stratum == stratum][
            ["horizon", "model", "mean_skill", "median_skill", "repo_win_rate", "n_repos"]
        ].round(3).to_string(index=False)
    )

print("\n" + "=" * 96)
print("LEAVE-ONE-REPOSITORY-OUT WINNER ROBUSTNESS")
print("=" * 96)
for (stratum, horizon), group in wide.groupby(level=["stratum", "horizon"]):
    full = group[MODELS].mean()
    winner = str(full.idxmin())
    flips = 0
    for repo_name in group.index.get_level_values("repo_name"):
        loo_winner = str(group.drop(index=(stratum, horizon, repo_name))[MODELS].mean().idxmin())
        flips += int(loo_winner != winner)
    print(
        f"{stratum:9s} h={int(horizon):2d}d winner={winner:8s} "
        f"winner flips after dropping one repo: {flips}/{len(group)}"
    )

print("\n" + "=" * 96)
print("FOUNDATION-MODEL GAIN CONCENTRATION VS BEST FIXED BASELINE PER CELL")
print("Best fixed baseline is selected by cell-level macro CRPS, not per window.")
print("=" * 96)
for (stratum, horizon), group in wide.groupby(level=["stratum", "horizon"]):
    base_means = group[BASELINES].mean()
    best_base = str(base_means.idxmin())
    for model in FOUNDATION:
        gain = group[best_base] - group[model]
        positive = gain.clip(lower=0)
        total_positive = float(positive.sum())
        top5_share = (
            float(positive.nlargest(min(5, len(positive))).sum() / total_positive)
            if total_positive > 0
            else float("nan")
        )
        print(
            f"{stratum:9s} h={int(horizon):2d}d {model:8s} vs {best_base:15s} "
            f"median gain={gain.median():7.3f} repo-win={(gain > 0).mean():5.1%} "
            f"top5 share of positive gain={top5_share:5.1%}"
        )

out = ROOT / "reports" / "review_scale_skill.csv"
skill_df.to_csv(out, index=False)
print(f"\nSaved {out}")
