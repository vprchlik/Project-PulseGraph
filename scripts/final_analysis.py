#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Final 6-model analysis + plots for the writeup.

Reads reports/experiment_real_full6_results.parquet (baselines + chronos2 +
timesfm). Produces:
  - CRPS / coverage / MAE tables by regime x horizon x model
  - paired win-rates for each foundation model vs the best baseline
  - supported 80% interval coverage by regime and horizon
  - example forecast spot-checks (one repo per regime)
All plots saved under reports/figures/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

res = pd.read_parquet(REPORTS / "experiment_real_full6_results.parquet")
MODELS = ["naive_seasonal", "linear", "ets", "lightgbm", "chronos2", "timesfm"]
BASELINES = ["naive_seasonal", "linear", "ets", "lightgbm"]
FOUNDATION = ["chronos2", "timesfm"]
REGIMES = ["breakout", "steady", "dead"]
HORIZONS = [7, 30, 90]

# ---------- tables ----------
def table(metric, agg="mean"):
    t = res.groupby(["stratum", "horizon", "model_name"])[metric].agg(agg).unstack("model_name")
    return t[MODELS].round(3)

print("=" * 92)
print("CRPS by regime x horizon x model  [lower = better]")
print("=" * 92)
print(table("crps_mean").to_string())
print("\n" + "=" * 92)
print("95% interval COVERAGE (nominal 0.95)")
print("=" * 92)
print(table("coverage_95").round(3).to_string())
print("\n" + "=" * 92)
print("80% interval COVERAGE (nominal 0.80)")
print("=" * 92)
print(table("coverage_80").round(3).to_string())
print("\n" + "=" * 92)
print("MAE (point/median) by regime x horizon x model  [lower = better]")
print("=" * 92)
print(table("mae").to_string())

# ---------- paired win-rates ----------
key = ["repo_name", "horizon", "cutoff_idx", "stratum"]
print("\n" + "=" * 92)
print("DESCRIPTIVE per-window win-rate vs BEST primary baseline (CRPS)")
print("Windows overlap; use review_cluster_inference.py for valid inference.")
print("=" * 92)
for fm in FOUNDATION:
    print(f"\n--- {fm} ---")
    fmdf = res[res.model_name == fm][key + ["crps_mean"]].rename(columns={"crps_mean": "crps_fm"})
    for regime in REGIMES:
        for h in HORIZONS:
            bb = (res[res.model_name.isin(BASELINES) & (res.horizon == h) & (res.stratum == regime)]
                  .groupby(key)["crps_mean"].min().reset_index()
                  .rename(columns={"crps_mean": "crps_bestbase"}))
            m = fmdf.merge(bb, on=key)
            m = m[(m.horizon == h) & (m.stratum == regime)]
            if m.empty:
                continue
            win = float((m.crps_fm < m.crps_bestbase).mean())
            print(f"  {regime:9s} h={h:2d}d  n={len(m):3d}  beats best-baseline in "
                  f"{win*100:5.1f}% of windows")

# ---------- supported interval calibration plot ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
for ax, regime in zip(axes, REGIMES):
    sub = res[res.stratum == regime]
    piv = (
        sub.groupby(["horizon", "model_name"])["coverage_80"]
        .mean()
        .unstack("model_name")[MODELS]
    )
    for model in MODELS:
        ax.plot(
            piv.index,
            piv[model],
            marker="o",
            linewidth=1.5,
            label=model,
        )
    ax.axhline(0.80, color="black", ls="--", lw=1, label="nominal 0.80")
    ax.set_title(regime)
    ax.set_xlabel("forecast horizon (days)")
    ax.set_xticks(HORIZONS)
    ax.set_ylim(0, 1)
    if regime == "breakout":
        ax.set_ylabel("empirical 80% interval coverage")
        ax.legend(fontsize=7, ncol=2)
fig.suptitle(
    "Supported 80% interval coverage by discovery cohort and horizon",
    fontsize=13,
)
fig.tight_layout()
fig.savefig(FIGS / "coverage80_by_regime.png", dpi=110)
plt.close(fig)
print(f"\nsaved {FIGS / 'coverage80_by_regime.png'}")

# ---------- primary-run CRPS-by-cohort bar chart ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
for ax, regime in zip(axes, REGIMES):
    sub = res[res.stratum == regime]
    piv = sub.groupby(["horizon", "model_name"])["crps_mean"].mean().unstack("model_name")[MODELS]
    piv.plot(kind="bar", ax=ax, width=0.8, legend=(regime == "breakout"))
    ax.set_title(f"{regime}")
    ax.set_xlabel("horizon (days)")
    ax.set_ylabel("mean CRPS")
    ax.tick_params(axis="x", rotation=0)
    if regime == "breakout":
        ax.legend(fontsize=7, ncol=2)
fig.suptitle(
    "Primary-run mean CRPS by discovery cohort, horizon, and model",
    fontsize=13,
)
fig.tight_layout()
fig.savefig(FIGS / "crps_by_regime.png", dpi=110)
plt.close(fig)
print(f"saved {FIGS / 'crps_by_regime.png'}")

# ---------- post-hoc strong-baseline sensitivity chart ----------
audit_path = REPORTS / "review_strong_baselines_results.parquet"
if audit_path.exists():
    audit = pd.read_parquet(audit_path)
    sensitivity = pd.concat([res, audit], ignore_index=True)
    selected = ["local_mean", "chronos2", "timesfm"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    for ax, regime in zip(axes, REGIMES):
        sub = sensitivity[
            (sensitivity.stratum == regime)
            & (sensitivity.model_name.isin(selected))
        ]
        piv = (
            sub.groupby(["horizon", "model_name"])["crps_mean"]
            .mean()
            .unstack("model_name")[selected]
        )
        for model in selected:
            ax.plot(piv.index, piv[model], marker="o", linewidth=2, label=model)
        ax.set_title(regime)
        ax.set_xlabel("forecast horizon (days)")
        ax.set_ylabel("mean CRPS")
        ax.set_xticks(HORIZONS)
        if regime == "breakout":
            ax.legend(fontsize=8)
    fig.suptitle(
        "Post-hoc sensitivity: foundation models vs 28-day local mean",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(FIGS / "crps_sensitivity_by_cohort.png", dpi=110)
    plt.close(fig)
    print(f"saved {FIGS / 'crps_sensitivity_by_cohort.png'}")

print("\nDONE")
