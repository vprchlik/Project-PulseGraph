#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Spot-check: plot trailing-window forecasts on one repo per discovery cohort.

This is a visual sanity check, not one of the primary early-life backtest
windows. Saves reports/figures/spotcheck_forecasts.png.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("TIMESFM_THREADS", "4")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import torch
torch.set_num_threads(4)

from pulsegraph.data.loader import load_daily_signals, prepare_star_series
from pulsegraph.forecast.chronos_forecaster import ChronosForecaster
from pulsegraph.forecast.timesfm_forecaster import TimesFMForecaster

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "reports" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

EXAMPLES = [
    ("breakout", "Tencent-Hunyuan/Hunyuan3D-2"),
    ("steady", "owasp-amass/amass"),
    ("dead", "dvdzkwsk/react-redux-starter-kit"),
]
HORIZON = 90

sig = load_daily_signals()
chronos = ChronosForecaster(n_samples=200)
timesfm = TimesFMForecaster(n_samples=200)

fig, axes = plt.subplots(3, 1, figsize=(12, 11))
for ax, (regime, repo) in zip(axes, EXAMPLES):
    s = prepare_star_series(sig, repo, "stars", 180)
    vals = s.values.astype(float)
    cutoff = len(vals) - HORIZON
    context = vals[:cutoff]
    actual = vals[cutoff:cutoff + HORIZON]

    ctx_tail = 120
    x_ctx = np.arange(-min(ctx_tail, len(context)), 0)
    x_fut = np.arange(0, HORIZON)

    ax.plot(x_ctx, context[-len(x_ctx):], color="#444", lw=1, label="context (history)")
    ax.plot(x_fut, actual, color="black", lw=1.8, label="actual")

    for fc, name, col in [
        (chronos.forecast(context, HORIZON, repo), "Chronos-2", "#7b3fbf"),
        (timesfm.forecast(context, HORIZON, repo), "TimesFM", "#b5651d"),
    ]:
        ax.plot(x_fut, fc.median, color=col, lw=1.6, label=f"{name} median")
        ax.fill_between(x_fut, fc.lower_80, fc.upper_80, color=col, alpha=0.18,
                        label=f"{name} 80% PI")

    ax.axvline(0, color="gray", ls=":", lw=1)
    ax.set_title(
        f"[{regime}-selected cohort] {repo} — trailing 90-day forecast",
        fontsize=11,
    )
    ax.set_xlabel("days relative to forecast cutoff")
    ax.set_ylabel("daily new stars")
    ax.legend(fontsize=7, ncol=3)

fig.suptitle(
    "Trailing-window forecast spot-check (separate from primary backtest)",
    fontsize=13,
)
fig.tight_layout()
out = FIGS / "spotcheck_forecasts.png"
fig.savefig(out, dpi=110)
plt.close(fig)
print("saved", out)
