#!/usr/bin/env python3
"""Run TimesFM-2.5 through the identical backtest harness/config as real_full
and merge into a 6-model result set (baselines + chronos2 + timesfm).

Reuses the exact dataset, windows, strata, and config from the real_full run so
TimesFM is evaluated on identical cutoffs. Avoids re-running the baselines and
Chronos-2 (already validated).
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from pulsegraph.config import RAW_DIR, REPORTS_DIR
from pulsegraph.data.loader import build_dataset_for_forecasting, load_daily_signals
from pulsegraph.evaluation.backtest import BacktestConfig, run_backtest_suite, summarize_backtest
from pulsegraph.evaluation.calibration import generate_calibration_report
from pulsegraph.forecast.timesfm_forecaster import TimesFMForecaster

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("add_timesfm")

# Config identical to the real_full run (from experiment_real_full_meta.json).
MIN_HISTORY_DAYS = 270
N_SAMPLES = 200
CONFIG = BacktestConfig(horizons=[7, 30, 90], step_days=30,
                        min_context_days=180, max_windows=8)


def main():
    meta = pd.read_parquet(RAW_DIR / "ingest_metadata.parquet")
    meta = meta[meta["status"] == "ok"]
    strata = dict(zip(meta["repo_name"], meta["regime_label"]))

    signals = load_daily_signals(RAW_DIR / "daily_signals.parquet")
    dataset = build_dataset_for_forecasting(
        signals, list(strata.keys()), signal_col="stars",
        min_history_days=MIN_HISTORY_DAYS,
    )
    logger.info("Dataset: %d repos", len(dataset))

    forecasters = {"timesfm": TimesFMForecaster(n_samples=N_SAMPLES)}
    tf_results = run_backtest_suite(dataset, forecasters, CONFIG, strata=strata)
    if tf_results.empty:
        logger.error("TimesFM produced no results. Aborting.")
        raise SystemExit(1)
    logger.info("TimesFM results: %d rows", len(tf_results))

    prev = pd.read_parquet(REPORTS_DIR / "experiment_real_full_results.parquet")
    combined = pd.concat([prev, tf_results], ignore_index=True)

    out = REPORTS_DIR / "experiment_real_full6_results.parquet"
    combined.to_parquet(out, index=False)
    summarize_backtest(combined).to_csv(
        REPORTS_DIR / "experiment_real_full6_summary.csv", index=False)
    generate_calibration_report(
        combined, output_dir=REPORTS_DIR / "calibration_real_full6")

    usable = {}
    for name in dataset:
        r = strata.get(name, "unknown")
        usable[r] = usable.get(r, 0) + 1
    meta_out = {
        "tag": "real_full6",
        "n_repos": len(dataset),
        "usable_by_regime": usable,
        "models": sorted(combined["model_name"].unique().tolist()),
        "config": {"horizons": CONFIG.horizons, "step_days": CONFIG.step_days,
                   "min_context_days": CONFIG.min_context_days,
                   "max_windows": CONFIG.max_windows, "n_samples": N_SAMPLES,
                   "min_history_days": MIN_HISTORY_DAYS},
    }
    with open(REPORTS_DIR / "experiment_real_full6_meta.json", "w") as f:
        json.dump(meta_out, f, indent=2)

    logger.info("Saved 6-model results -> %s", out)
    logger.info("Models: %s", meta_out["models"])
    logger.info("Rows: %d", len(combined))


if __name__ == "__main__":
    main()
