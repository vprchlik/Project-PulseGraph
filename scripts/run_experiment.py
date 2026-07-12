#!/usr/bin/env python3
"""Main experiment: foundation models vs baselines on real GitHub star series.

Answers: do time-series foundation models (Chronos-2, TimesFM) beat simple
baselines at forecasting shock-driven attention signals, at which horizons,
and for which series types (breakout / steady / dead)?

Models are added incrementally and verified one at a time:
    # 1. baselines only (default)
    python scripts/run_experiment.py
    # 2. add Chronos-2
    python scripts/run_experiment.py --include-chronos
    # 3. add TimesFM
    python scripts/run_experiment.py --include-chronos --include-timesfm

Stratification is by REGIME (breakout/steady/dead) from the ingestion
metadata, not by star-count tiers.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pulsegraph.config import RAW_DIR, REPORTS_DIR
from pulsegraph.data.loader import build_dataset_for_forecasting, load_daily_signals
from pulsegraph.evaluation.backtest import (
    BacktestConfig,
    run_backtest_suite,
    summarize_backtest,
)
from pulsegraph.evaluation.calibration import generate_calibration_report
from pulsegraph.forecast.baselines import BASELINE_FORECASTERS

PRIMARY_BASELINES = ["naive_seasonal", "linear", "ets", "lightgbm"]
AUDIT_BASELINES = ["last_value", "weekly_naive", "local_mean"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("experiment")


def load_regime_strata() -> dict[str, str]:
    """Load repo_name -> regime label from ingestion metadata."""
    meta_path = RAW_DIR / "ingest_metadata.parquet"
    if not meta_path.exists():
        meta_path = RAW_DIR / "target_repos.parquet"
    meta = pd.read_parquet(meta_path)
    if "status" in meta.columns:
        meta = meta[meta["status"] == "ok"]
    return dict(zip(meta["repo_name"], meta["regime_label"]))


def summarize_by_regime(results: pd.DataFrame) -> pd.DataFrame:
    """CRPS / coverage per model x horizon x regime (stratum)."""
    return summarize_backtest(results)


def main():
    parser = argparse.ArgumentParser(description="Foundation models vs baselines")
    parser.add_argument("--data-path", default=str(RAW_DIR / "daily_signals.parquet"))
    parser.add_argument("--min-history-days", type=int, default=270,
                        help="Min series length (>=180 context + 90 horizon)")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--max-windows", type=int, default=8)
    parser.add_argument("--step-days", type=int, default=30)
    parser.add_argument("--min-context-days", type=int, default=180)
    parser.add_argument("--include-chronos", action="store_true")
    parser.add_argument("--include-timesfm", action="store_true")
    parser.add_argument(
        "--include-audit-baselines",
        action="store_true",
        help="Add deterministic last-value, weekly-naive, and local-mean sensitivity baselines",
    )
    parser.add_argument("--tag", default="real", help="Report filename tag")
    args = parser.parse_args()

    signals = load_daily_signals(Path(args.data_path))
    strata = load_regime_strata()

    repo_names = list(strata.keys())
    logger.info("Loaded %d repos with regime labels", len(repo_names))

    dataset = build_dataset_for_forecasting(
        signals, repo_names, signal_col="stars", min_history_days=args.min_history_days
    )
    if not dataset:
        logger.error("No repos with sufficient history. Aborting.")
        sys.exit(1)

    # Report usable counts per regime.
    usable_by_regime: dict[str, int] = {}
    for name in dataset:
        r = strata.get(name, "unknown")
        usable_by_regime[r] = usable_by_regime.get(r, 0) + 1
    logger.info("Usable repos by regime: %s", usable_by_regime)

    forecasters: dict = {}
    baseline_names = PRIMARY_BASELINES + (
        AUDIT_BASELINES if args.include_audit_baselines else []
    )
    for name in baseline_names:
        cls = BASELINE_FORECASTERS[name]
        forecasters[name] = cls(n_samples=args.n_samples)

    if args.include_chronos:
        from pulsegraph.forecast.chronos_forecaster import ChronosForecaster
        forecasters["chronos2"] = ChronosForecaster(n_samples=args.n_samples)
        logger.info("Chronos-2 added")

    if args.include_timesfm:
        from pulsegraph.forecast.timesfm_forecaster import TimesFMForecaster
        forecasters["timesfm"] = TimesFMForecaster(n_samples=args.n_samples)
        logger.info("TimesFM added")

    config = BacktestConfig(
        horizons=[7, 30, 90],
        step_days=args.step_days,
        min_context_days=args.min_context_days,
        max_windows=args.max_windows,
    )

    logger.info("Backtest: %d repos x %d models, horizons=%s, max_windows=%d",
                len(dataset), len(forecasters), config.horizons, config.max_windows)
    results = run_backtest_suite(dataset, forecasters, config, strata=strata)

    if results.empty:
        logger.error("No results produced. Aborting.")
        sys.exit(1)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = REPORTS_DIR / f"experiment_{args.tag}_results.parquet"
    results.to_parquet(results_path, index=False)
    logger.info("Saved raw results -> %s", results_path)

    summary = summarize_backtest(results)
    summary_path = REPORTS_DIR / f"experiment_{args.tag}_summary.csv"
    summary.to_csv(summary_path, index=False)

    cal = generate_calibration_report(results,
                                      output_dir=REPORTS_DIR / f"calibration_{args.tag}")

    meta = {
        "tag": args.tag,
        "n_repos": len(dataset),
        "usable_by_regime": usable_by_regime,
        "models": list(forecasters.keys()),
        "config": {
            "horizons": config.horizons,
            "step_days": config.step_days,
            "min_context_days": config.min_context_days,
            "max_windows": config.max_windows,
            "n_samples": args.n_samples,
        },
    }
    with open(REPORTS_DIR / f"experiment_{args.tag}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print_summary(summary, cal)


def print_summary(summary: pd.DataFrame, cal: dict):
    print("\n" + "=" * 78)
    print("  EXPERIMENT SUMMARY — CRPS by model x horizon x regime (lower=better)")
    print("=" * 78)
    # Aggregate across regimes for a headline table.
    for horizon in sorted(summary["horizon"].unique()):
        h = summary[summary["horizon"] == horizon]
        agg = h.groupby("model_name").apply(
            lambda g: (g["crps_mean"] * g["n_windows"]).sum() / g["n_windows"].sum()
        ).sort_values()
        print(f"\n  Horizon {horizon}d (windows-weighted CRPS):")
        for model, crps in agg.items():
            print(f"    {model:16s} {crps:8.4f}")
    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
