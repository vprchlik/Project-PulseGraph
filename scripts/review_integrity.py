#!/usr/bin/env python3
"""Review audit 1: run integrity of the 6-model results.

Checks:
  1. Every model was evaluated on the *identical* set of (repo, horizon,
     cutoff_idx) windows (no stitched/partial runs).
  2. The summary CSV matches recomputation from the raw parquet.
  3. The numbers cited in RESULTS.md tables match the parquet.
  4. cutoff_date is consistent with cutoff_idx for every row.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

res = pd.read_parquet(REPORTS / "experiment_real_full6_results.parquet")
print(f"rows={len(res)}  models={sorted(res.model_name.unique())}")
print(f"columns={list(res.columns)}")

# --- 1. identical windows across models -------------------------------------
key = ["repo_name", "horizon", "cutoff_idx"]
sets = {}
for m, g in res.groupby("model_name"):
    sets[m] = set(map(tuple, g[key].values))
ref_model = "naive_seasonal"
ok = True
for m, s in sets.items():
    if s != sets[ref_model]:
        ok = False
        print(f"  MISMATCH {m}: only-in-{m}={len(s - sets[ref_model])} "
              f"missing={len(sets[ref_model] - s)}")
print(f"[1] identical windows across all 6 models: {'PASS' if ok else 'FAIL'}")

dup = res.duplicated(subset=key + ["model_name"]).sum()
print(f"[1b] duplicate (window, model) rows: {dup} {'PASS' if dup == 0 else 'FAIL'}")

# stratum consistent per repo?
s_per_repo = res.groupby("repo_name")["stratum"].nunique()
print(f"[1c] repos with >1 stratum label: {(s_per_repo > 1).sum()} "
      f"{'PASS' if (s_per_repo > 1).sum() == 0 else 'FAIL'}")

# --- 2. summary CSV matches recomputation -----------------------------------
summ = pd.read_csv(REPORTS / "experiment_real_full6_summary.csv")
recomp = res.groupby(["model_name", "horizon", "stratum"]).agg(
    crps_mean=("crps_mean", "mean"),
    crps_std=("crps_mean", "std"),
    mape_mean=("mape", "mean"),
    mape_median=("mape", "median"),
    mae_mean=("mae", "mean"),
    coverage_80_mean=("coverage_80", "mean"),
    coverage_95_mean=("coverage_95", "mean"),
    n_windows=("crps_mean", "count"),
).reset_index()
merged = summ.merge(recomp, on=["model_name", "horizon", "stratum"],
                    suffixes=("_csv", "_re"))
bad = 0
for col in [
    "crps_mean",
    "crps_std",
    "mape_mean",
    "mape_median",
    "mae_mean",
    "coverage_80_mean",
    "coverage_95_mean",
    "n_windows",
]:
    d = np.abs(merged[f"{col}_csv"] - merged[f"{col}_re"])
    if d.max() > 1e-9:
        bad += 1
        print(f"  {col}: max abs diff {d.max():.3e} FAIL")
print(f"[2] summary CSV == recomputation from parquet: {'PASS' if bad == 0 else 'FAIL'} "
      f"({len(merged)} cells checked)")

# --- 3. RESULTS.md CRPS table spot-verified ---------------------------------
results_tbl = {  # (regime, h, model): value in RESULTS.md §4.1
    ("breakout", 7): dict(naive_seasonal=11.82, linear=11.19, ets=12.59,
                          lightgbm=14.82, chronos2=6.43, timesfm=6.23),
    ("breakout", 30): dict(naive_seasonal=13.32, linear=16.69, ets=18.11,
                           lightgbm=19.40, chronos2=7.90, timesfm=7.68),
    ("breakout", 90): dict(naive_seasonal=14.55, linear=28.00, ets=23.99,
                           lightgbm=23.33, chronos2=8.52, timesfm=8.20),
    ("steady", 7): dict(naive_seasonal=2.50, linear=2.41, ets=2.11,
                        lightgbm=4.39, chronos2=1.46, timesfm=1.44),
    ("steady", 30): dict(naive_seasonal=3.30, linear=3.99, ets=2.83,
                         lightgbm=5.64, chronos2=2.04, timesfm=2.02),
    ("steady", 90): dict(naive_seasonal=3.29, linear=6.41, ets=3.07,
                         lightgbm=6.15, chronos2=2.21, timesfm=2.22),
    ("dead", 7): dict(naive_seasonal=3.12, linear=2.82, ets=3.02,
                      lightgbm=5.10, chronos2=2.43, timesfm=2.35),
    ("dead", 30): dict(naive_seasonal=3.96, linear=3.66, ets=3.42,
                       lightgbm=6.21, chronos2=2.47, timesfm=2.39),
    ("dead", 90): dict(naive_seasonal=4.10, linear=5.68, ets=3.75,
                       lightgbm=7.46, chronos2=2.73, timesfm=2.67),
}
mae_tbl = {
    ("breakout", 7): dict(naive_seasonal=14.04, linear=14.49, ets=14.44,
                          lightgbm=15.56, chronos2=8.29, timesfm=8.00),
    ("breakout", 90): dict(naive_seasonal=17.25, linear=31.51, ets=26.17,
                           lightgbm=24.10, chronos2=10.20, timesfm=9.78),
    ("steady", 7): dict(naive_seasonal=3.02, linear=3.04, ets=2.60,
                        lightgbm=4.59, chronos2=1.82, timesfm=1.80),
    ("dead", 90): dict(naive_seasonal=5.00, linear=6.57, ets=4.59,
                       lightgbm=7.75, chronos2=3.26, timesfm=3.25),
}
n_bad = 0
for (regime, h), row in results_tbl.items():
    for model, doc_val in row.items():
        actual = res[(res.stratum == regime) & (res.horizon == h)
                     & (res.model_name == model)]["crps_mean"].mean()
        if abs(round(actual, 2) - doc_val) > 0.005:
            n_bad += 1
            print(f"  CRPS {regime} h{h} {model}: doc={doc_val} actual={actual:.4f} FAIL")
for (regime, h), row in mae_tbl.items():
    for model, doc_val in row.items():
        actual = res[(res.stratum == regime) & (res.horizon == h)
                     & (res.model_name == model)]["mae"].mean()
        if abs(round(actual, 2) - doc_val) > 0.005:
            n_bad += 1
            print(f"  MAE {regime} h{h} {model}: doc={doc_val} actual={actual:.4f} FAIL")
status = "PASS" if n_bad == 0 else f"FAIL ({n_bad} cells)"
print(f"[3] RESULTS.md CRPS+MAE tables vs parquet: {status}")

# Coverage claims retained in RESULTS.md §4.4.
cov = res.groupby(["model_name", "stratum", "horizon"])["coverage_95"].mean().reset_index()
chronos_95 = cov[cov.model_name == "chronos2"]["coverage_95"]
print(
    "\n[3b] Chronos supported 95% coverage range: "
    f"{chronos_95.min():.3f}-{chronos_95.max():.3f}"
)
cov80 = res.groupby(["model_name", "stratum", "horizon"])["coverage_80"].mean().reset_index()
print("[3c] coverage_80 overall min-max:")
for m in ["chronos2", "timesfm"]:
    v = cov80[cov80.model_name == m]["coverage_80"]
    print(f"  {m:15s} {v.min():.3f}-{v.max():.3f}")

# Post-hoc local-mean sensitivity values in RESULTS.md §4.2.
audit = pd.read_parquet(REPORTS / "review_strong_baselines_results.parquet")
local_expected = {
    ("breakout", 7): 7.77,
    ("breakout", 30): 8.77,
    ("breakout", 90): 9.41,
    ("steady", 7): 1.66,
    ("steady", 30): 2.26,
    ("steady", 90): 2.42,
    ("dead", 7): 2.75,
    ("dead", 30): 2.64,
    ("dead", 90): 2.85,
}
local_bad = 0
for (regime, horizon), expected in local_expected.items():
    actual = audit[
        (audit.stratum == regime)
        & (audit.horizon == horizon)
        & (audit.model_name == "local_mean")
    ]["crps_mean"].mean()
    local_bad += int(abs(round(actual, 2) - expected) > 0.005)
print(
    f"[3d] RESULTS.md local-mean sensitivity table: "
    f"{'PASS' if local_bad == 0 else f'FAIL ({local_bad} cells)'}"
)

# --- 4. cutoff_date vs cutoff_idx consistency -------------------------------
# cutoff_idx should map to the same cutoff_date within a repo.
chk = res.groupby(["repo_name", "cutoff_idx"])["cutoff_date"].nunique()
print(f"\n[4] (repo, cutoff_idx) with inconsistent cutoff_date: {(chk > 1).sum()} "
      f"{'PASS' if (chk > 1).sum() == 0 else 'FAIL'}")

# n_windows per horizon: same windows for all horizons?
per_h = res[res.model_name == ref_model].groupby("horizon").size()
print(f"[5] windows per horizon (should be equal): {per_h.to_dict()}")
