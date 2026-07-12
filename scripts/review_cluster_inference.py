#!/usr/bin/env python3
"""Repository-clustered inference for foundation models vs local mean.

Forecast windows overlap (especially at h=90), so treating every window as an
independent Wilcoxon observation is anti-conservative. This sensitivity check
first averages CRPS within each repository, then performs paired inference
across repositories. It reports:

* repository win rate;
* mean and median paired CRPS difference (FM - local mean);
* a repository bootstrap 95% CI for the mean difference;
* a paired Wilcoxon p-value at repository level; and
* Benjamini-Hochberg adjusted p-values across all 18 comparisons.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
PRIMARY = pd.read_parquet(ROOT / "reports" / "experiment_real_full6_results.parquet")
AUDIT = pd.read_parquet(ROOT / "reports" / "review_strong_baselines_results.parquet")
res = pd.concat([PRIMARY, AUDIT], ignore_index=True)

repo = (
    res[res["model_name"].isin(["local_mean", "chronos2", "timesfm"])]
    .groupby(["stratum", "horizon", "repo_name", "model_name"])["crps_mean"]
    .mean()
    .unstack("model_name")
)

rng = np.random.default_rng(20260712)
rows: list[dict] = []
for (stratum, horizon), group in repo.groupby(level=["stratum", "horizon"]):
    for model in ["chronos2", "timesfm"]:
        diff = (group[model] - group["local_mean"]).to_numpy(dtype=float)
        n = len(diff)
        boot_idx = rng.integers(0, n, size=(20_000, n))
        boot_means = diff[boot_idx].mean(axis=1)
        try:
            _, p = stats.wilcoxon(diff)
        except ValueError:
            p = float("nan")
        rows.append(
            {
                "stratum": stratum,
                "horizon": int(horizon),
                "model": model,
                "n_repos": n,
                "repo_win_rate": float((diff < 0).mean()),
                "mean_diff": float(diff.mean()),
                "median_diff": float(np.median(diff)),
                "mean_diff_ci_low": float(np.quantile(boot_means, 0.025)),
                "mean_diff_ci_high": float(np.quantile(boot_means, 0.975)),
                "wilcoxon_p": float(p),
            }
        )

out = pd.DataFrame(rows)

# Benjamini-Hochberg false-discovery-rate adjustment.
pvals = out["wilcoxon_p"].to_numpy()
order = np.argsort(pvals)
ranked = pvals[order] * len(pvals) / np.arange(1, len(pvals) + 1)
ranked = np.minimum.accumulate(ranked[::-1])[::-1]
adjusted = np.empty_like(ranked)
adjusted[order] = np.minimum(ranked, 1.0)
out["wilcoxon_p_bh"] = adjusted

print("=" * 106)
print("REPOSITORY-CLUSTERED CRPS INFERENCE: FM - local_mean (negative favors FM)")
print("=" * 106)
print(
    out.round(
        {
            "repo_win_rate": 3,
            "mean_diff": 3,
            "median_diff": 3,
            "mean_diff_ci_low": 3,
            "mean_diff_ci_high": 3,
            "wilcoxon_p": 5,
            "wilcoxon_p_bh": 5,
        }
    ).to_string(index=False)
)

path = ROOT / "reports" / "review_cluster_inference.csv"
out.to_csv(path, index=False)
print(f"\nSaved {path}")
