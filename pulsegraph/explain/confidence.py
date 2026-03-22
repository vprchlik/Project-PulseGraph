"""Confidence scoring and abstention logic.

Computes a confidence score based on signal quality and triggers
abstention when the model is under-informed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


ABSTENTION_THRESHOLD = 0.3


def compute_confidence(
    repo_name: str,
    series: pd.Series,
    min_history_days: int = 90,
    min_nonzero_frac: float = 0.3,
) -> dict:
    """Compute confidence score and abstention decision for a repo.

    Score components:
    - history_score: longer history -> higher confidence
    - stability_score: lower relative volatility -> higher confidence
    - completeness_score: fewer zero/missing days -> higher confidence
    """
    values = series.values.astype(np.float64)
    n = len(values)

    history_score = min(1.0, n / 365.0)

    mean_val = np.mean(values) if np.mean(values) > 0 else 1.0
    cv = np.std(values) / mean_val
    stability_score = max(0.0, 1.0 - cv / 3.0)

    nonzero_frac = np.count_nonzero(values) / max(n, 1)
    completeness_score = min(1.0, nonzero_frac / 0.8)

    confidence = (
        0.4 * history_score
        + 0.3 * stability_score
        + 0.3 * completeness_score
    )

    abstain = False
    abstain_reason = ""

    if n < min_history_days:
        abstain = True
        abstain_reason = f"Insufficient history ({n} days < {min_history_days} minimum)"
    elif nonzero_frac < min_nonzero_frac:
        abstain = True
        abstain_reason = f"Too sparse ({nonzero_frac:.0%} non-zero < {min_nonzero_frac:.0%} minimum)"
    elif confidence < ABSTENTION_THRESHOLD:
        abstain = True
        abstain_reason = f"Low confidence ({confidence:.2f} < {ABSTENTION_THRESHOLD})"

    return {
        "repo_name": repo_name,
        "confidence_score": float(confidence),
        "signal_history_days": n,
        "signal_volatility": float(cv),
        "data_completeness": float(nonzero_frac),
        "abstain": abstain,
        "abstain_reason": abstain_reason,
    }
