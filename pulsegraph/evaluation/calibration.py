"""Calibration analysis: PIT histograms, reliability diagrams, Brier scores."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pulsegraph.config import REPORTS_DIR

logger = logging.getLogger(__name__)


def compute_pit_histogram(
    pit_values: list[np.ndarray],
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate PIT values across multiple forecasts and compute histogram.

    Returns (bin_counts_normalized, bin_edges).
    Ideal calibration: all bins equal to 1/n_bins.
    """
    all_pit = np.concatenate(pit_values)
    counts, edges = np.histogram(all_pit, bins=n_bins, range=(0, 1))
    normalized = counts / counts.sum()
    return normalized, edges


def reliability_diagram_data(
    predicted_probs: np.ndarray,
    observed_outcomes: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute data for a reliability diagram.

    Groups predictions into bins by predicted probability and computes
    the observed frequency in each bin.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(predicted_probs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    records = []
    for b in range(n_bins):
        mask = bin_indices == b
        if np.sum(mask) == 0:
            continue
        records.append({
            "bin_center": (bins[b] + bins[b + 1]) / 2,
            "predicted_mean": float(np.mean(predicted_probs[mask])),
            "observed_freq": float(np.mean(observed_outcomes[mask])),
            "count": int(np.sum(mask)),
        })

    return pd.DataFrame(records)


def brier_score(predicted_probs: np.ndarray, observed_outcomes: np.ndarray) -> float:
    """Brier score: mean squared error between predicted probabilities and binary outcomes."""
    return float(np.mean((predicted_probs - observed_outcomes) ** 2))


def generate_calibration_report(
    backtest_results: pd.DataFrame,
    output_dir: Path | None = None,
) -> dict:
    """Generate calibration analysis from backtest results.

    Expects backtest_results to have a 'pit_values' column with lists of floats.
    """
    if output_dir is None:
        output_dir = REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {}

    if "pit_values" not in backtest_results.columns:
        logger.warning("No pit_values in backtest results, skipping PIT analysis")
        return report

    for model_name in backtest_results["model_name"].unique():
        model_data = backtest_results[backtest_results["model_name"] == model_name]

        for horizon in model_data["horizon"].unique():
            horizon_data = model_data[model_data["horizon"] == horizon]

            pit_lists = horizon_data["pit_values"].tolist()
            pit_arrays = [np.array(p) for p in pit_lists if len(p) > 0]

            if not pit_arrays:
                continue

            hist, edges = compute_pit_histogram(pit_arrays)

            key = f"{model_name}_h{horizon}"
            report[key] = {
                "model": model_name,
                "horizon": horizon,
                "pit_histogram": hist.tolist(),
                "pit_bin_edges": edges.tolist(),
                "pit_uniformity_score": float(np.std(hist)),
                "n_forecasts": len(pit_arrays),
            }

    report_df = pd.DataFrame(
        [v for v in report.values() if isinstance(v, dict)]
    )
    if not report_df.empty:
        report_path = output_dir / "calibration_report.csv"
        report_df.to_csv(report_path, index=False)
        logger.info("Calibration report saved to %s", report_path)

    return report
