#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Controlled test: does the quantile-grid vs random-sample representation of a
predictive distribution give an unfair empirical-CRPS advantage?

Chronos-2 pseudo-samples come from inverse-CDF interpolation of 21 native
quantiles (a smooth, evenly-spaced-probability grid). Baselines use 200 random
bootstrap samples. If, for the *same* underlying distribution, the quantile-grid
representation systematically yields lower empirical CRPS, part of Chronos's edge
would be an artifact rather than skill.

We compare both representations against the analytic CRPS of a known Normal, and
against each other, across many trials.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pulsegraph.evaluation.metrics import crps_empirical

rng = np.random.default_rng(0)

NATIVE_LEVELS = np.array([0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                          0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85,
                          0.90, 0.95, 0.99])
N = 200
sample_probs = np.linspace(0.5 / N, 1 - 0.5 / N, N)


def analytic_crps_normal(mu, sigma, y):
    z = (y - mu) / sigma
    return sigma * (z * (2 * stats.norm.cdf(z) - 1)
                    + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))


def quantile_grid_samples(mu, sigma):
    # Emulate Chronos path: get native quantiles of the SAME distribution,
    # then inverse-CDF interpolate to N pseudo-samples.
    qvals = stats.norm.ppf(NATIVE_LEVELS, loc=mu, scale=sigma)
    return np.interp(sample_probs, NATIVE_LEVELS, qvals)


n_trials = 4000
err_boot, err_grid = [], []
crps_boot_list, crps_grid_list = [], []
for _ in range(n_trials):
    mu = rng.uniform(-5, 5)
    sigma = rng.uniform(0.5, 5)
    y = mu + rng.normal() * sigma  # observation from same dist
    analytic = analytic_crps_normal(mu, sigma, y)

    boot = rng.normal(mu, sigma, size=N)          # baseline-style random samples
    grid = quantile_grid_samples(mu, sigma)        # chronos-style quantile grid

    c_boot = crps_empirical(boot, y)
    c_grid = crps_empirical(grid, y)
    crps_boot_list.append(c_boot)
    crps_grid_list.append(c_grid)
    err_boot.append(c_boot - analytic)
    err_grid.append(c_grid - analytic)

err_boot = np.array(err_boot)
err_grid = np.array(err_grid)
cb = np.array(crps_boot_list)
cg = np.array(crps_grid_list)

print("=" * 74)
print("CONTROLLED CRPS REPRESENTATION TEST (same Normal distribution, N=200)")
print(f"trials: {n_trials}")
print("=" * 74)
print(f"mean bias vs analytic CRPS  | bootstrap samples: {err_boot.mean():+.5f}")
print(f"                            | quantile grid     : {err_grid.mean():+.5f}")
print(f"RMSE vs analytic            | bootstrap samples: {np.sqrt((err_boot**2).mean()):.5f}")
print(f"                            | quantile grid     : {np.sqrt((err_grid**2).mean()):.5f}")
print()
# Does the grid systematically score lower for the SAME distribution?
diff = cg - cb
print(f"mean(CRPS_grid - CRPS_boot) : {diff.mean():+.5f}  "
      f"(as % of mean CRPS: {100*diff.mean()/cb.mean():+.2f}%)")
print(f"fraction of trials grid < boot: {(cg < cb).mean()*100:.1f}%")
print()
print("Interpretation: if mean bias and the grid-vs-boot gap are both ~0, the")
print("representation is NOT the source of Chronos's advantage; observed CRPS")
print("differences reflect genuinely different predictive distributions.")
