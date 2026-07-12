#!/usr/bin/env python3
"""Phase 0: Feasibility Validation

Answers the question: "Do foundation models actually help for this domain?"

This script:
1. Loads daily star data (from BigQuery backfill or synthetic fallback)
2. Runs Chronos-2 zero-shot and four baselines on held-out test windows
3. Computes CRPS and MAPE at 7, 30, 90-day horizons
4. Stratifies results by repo popularity
5. Produces a go/no-go report

Usage:
    # With real GH Archive data (requires BigQuery credentials):
    python scripts/phase0_feasibility.py --data-source bigquery

    # With synthetic data (no credentials needed, for development):
    python scripts/phase0_feasibility.py --data-source synthetic

    # With a local parquet file:
    python scripts/phase0_feasibility.py --data-source file \
        --data-path data/raw/daily_signals.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pulsegraph.config import RAW_DIR, REPORTS_DIR
from pulsegraph.data.loader import (
    build_dataset_for_forecasting,
    generate_synthetic_data,
    load_daily_signals,
)
from pulsegraph.evaluation.backtest import (
    BacktestConfig,
    run_backtest_suite,
    summarize_backtest,
)
from pulsegraph.evaluation.calibration import generate_calibration_report
from pulsegraph.forecast.baselines import BASELINE_FORECASTERS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("phase0")


def load_data(args) -> pd.DataFrame:
    """Load time-series data based on the chosen source."""
    if args.data_source == "synthetic":
        logger.info("Generating synthetic data for feasibility testing")
        df = generate_synthetic_data(
            n_repos=args.n_repos,
            n_days=365 * 3,
            seed=42,
        )
        out_path = RAW_DIR / "synthetic_daily_signals.parquet"
        df.to_parquet(out_path, index=False)
        logger.info("Saved synthetic data to %s", out_path)
        return df

    elif args.data_source == "bigquery":
        from pulsegraph.data.gharchive import GHArchiveClient, build_target_repo_list

        logger.info("Building target repo list from GH Archive")
        repos_df = build_target_repo_list(
            top_n=min(args.n_repos, 1000),
            tail_n=min(args.n_repos // 2, 500),
        )
        repo_names = repos_df["repo_name"].tolist()

        client = GHArchiveClient()
        out_path = RAW_DIR / "daily_signals.parquet"
        client.backfill_to_parquet(repo_names, output_path=out_path)
        return load_daily_signals(out_path)

    elif args.data_source == "file":
        data_path = Path(args.data_path)
        logger.info("Loading data from %s", data_path)
        return load_daily_signals(data_path)

    else:
        raise ValueError(f"Unknown data source: {args.data_source}")


def assign_strata(signals_df: pd.DataFrame) -> dict[str, str]:
    """Assign repos to strata based on total star count."""
    totals = signals_df.groupby("repo_name")["stars"].sum()

    strata = {}
    for repo, total in totals.items():
        if total >= 5000:
            strata[repo] = "high"
        elif total >= 1000:
            strata[repo] = "mid"
        else:
            strata[repo] = "low"
    return strata


def run_feasibility(args):
    """Main feasibility study pipeline."""
    signals_df = load_data(args)

    repo_names = signals_df["repo_name"].unique().tolist()
    if args.n_eval_repos and args.n_eval_repos < len(repo_names):
        rng = np.random.default_rng(42)
        repo_names = list(rng.choice(repo_names, size=args.n_eval_repos, replace=False))

    logger.info("Building forecasting dataset for %d repos", len(repo_names))
    dataset = build_dataset_for_forecasting(
        signals_df, repo_names, signal_col="stars", min_history_days=180
    )

    if not dataset:
        logger.error("No repos with sufficient history. Aborting.")
        return

    strata = assign_strata(signals_df)

    forecasters: dict = {}
    for name, cls in BASELINE_FORECASTERS.items():
        forecasters[name] = cls(n_samples=args.n_samples)

    if not args.baselines_only:
        try:
            from pulsegraph.forecast.chronos_forecaster import ChronosForecaster
            forecasters["chronos2"] = ChronosForecaster(n_samples=args.n_samples)
            logger.info("Chronos-2 loaded successfully")
        except Exception as e:
            logger.warning("Failed to load Chronos-2, running baselines only: %s", e)

    config = BacktestConfig(
        horizons=[7, 30, 90],
        step_days=14,
        min_context_days=180,
        max_windows=args.max_windows,
    )

    logger.info(
        "Starting backtest: %d repos, %d models, horizons=%s",
        len(dataset), len(forecasters), config.horizons,
    )
    results = run_backtest_suite(dataset, forecasters, config, strata=strata)

    if results.empty:
        logger.error("No backtest results produced. Aborting.")
        return

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results_path = REPORTS_DIR / "phase0_backtest_results.parquet"
    results.to_parquet(results_path, index=False)
    logger.info("Saved raw results to %s", results_path)

    summary = summarize_backtest(results)
    summary_path = REPORTS_DIR / "phase0_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("Saved summary to %s", summary_path)

    cal_report = generate_calibration_report(results)

    report = generate_go_nogo_report(summary, cal_report)
    report_path = REPORTS_DIR / "phase0_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Saved go/no-go report to %s", report_path)

    print_report(report)


def generate_go_nogo_report(summary: pd.DataFrame, calibration: dict) -> dict:
    """Analyze results and produce a go/no-go recommendation."""
    report = {
        "title": "Phase 0 Feasibility Report",
        "summary_table": summary.to_dict(orient="records"),
        "findings": [],
        "recommendation": "",
    }

    if summary.empty:
        report["recommendation"] = "INSUFFICIENT DATA"
        return report

    models = summary["model_name"].unique()
    has_chronos = "chronos2" in models

    if not has_chronos:
        report["findings"].append(
            "Chronos-2 was not available. Only baseline comparison was performed."
        )

    for horizon in sorted(summary["horizon"].unique()):
        h_data = summary[summary["horizon"] == horizon]
        best_model = h_data.loc[h_data["crps_mean"].idxmin(), "model_name"]
        best_crps = h_data["crps_mean"].min()

        report["findings"].append(
            f"Horizon {horizon}d: best model = {best_model} (CRPS={best_crps:.4f})"
        )

        if has_chronos:
            chronos_row = h_data[h_data["model_name"] == "chronos2"]
            if not chronos_row.empty:
                chronos_crps = chronos_row["crps_mean"].values[0]
                ets_row = h_data[h_data["model_name"] == "ets"]
                lgbm_row = h_data[h_data["model_name"] == "lightgbm"]

                beats_ets = not ets_row.empty and chronos_crps < ets_row["crps_mean"].values[0]
                beats_lgbm = not lgbm_row.empty and chronos_crps < lgbm_row["crps_mean"].values[0]

                if beats_ets and beats_lgbm:
                    report["findings"].append(
                        f"  -> Chronos-2 beats both ETS and LightGBM at {horizon}d. GO."
                    )
                elif beats_ets or beats_lgbm:
                    report["findings"].append(
                        f"  -> Chronos-2 beats one baseline at {horizon}d. CONDITIONAL GO."
                    )
                else:
                    report["findings"].append(
                        f"  -> Chronos-2 does NOT beat baselines at {horizon}d. REASSESS."
                    )

    if has_chronos:
        chronos_wins = sum(1 for f in report["findings"] if "GO." in f and "REASSESS" not in f)
        total_horizons = len(summary["horizon"].unique())
        if chronos_wins >= total_horizons:
            report["recommendation"] = "GO: Chronos-2 outperforms baselines across horizons."
        elif chronos_wins > 0:
            report["recommendation"] = (
                "CONDITIONAL GO: Chronos-2 shows advantage on some horizons. "
                "Consider fine-tuning or ensemble for weak horizons."
            )
        else:
            report["recommendation"] = (
                "REASSESS: Chronos-2 does not outperform baselines. "
                "Options: (a) fine-tune on GitHub data, (b) use domain-specific primary model, "
                "(c) proceed with regime search as core value."
            )
    else:
        report["recommendation"] = (
            "BASELINES ONLY: Install chronos-forecasting and GPU support "
            "to complete the feasibility comparison."
        )

    return report


def print_report(report: dict):
    """Pretty-print the go/no-go report."""
    print("\n" + "=" * 70)
    print(f"  {report['title']}")
    print("=" * 70)
    for finding in report.get("findings", []):
        print(f"  {finding}")
    print("-" * 70)
    print(f"  RECOMMENDATION: {report['recommendation']}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Phase 0: Feasibility Validation")
    parser.add_argument(
        "--data-source",
        choices=["synthetic", "bigquery", "file"],
        default="synthetic",
        help="Where to load data from",
    )
    parser.add_argument(
        "--data-path",
        default="",
        help="Path to data file (for --data-source=file)",
    )
    parser.add_argument("--n-repos", type=int, default=100, help="Number of repos to generate/pull")
    parser.add_argument("--n-eval-repos", type=int, default=50, help="Repos to evaluate (subset)")
    parser.add_argument("--n-samples", type=int, default=200, help="Forecast trajectory samples")
    parser.add_argument("--max-windows", type=int, default=10, help="Max backtest windows per repo")
    parser.add_argument("--baselines-only", action="store_true", help="Skip Chronos-2")

    args = parser.parse_args()
    run_feasibility(args)


if __name__ == "__main__":
    main()
