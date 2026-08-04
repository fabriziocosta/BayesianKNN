"""Monte Carlo convergence diagnostics."""

from __future__ import annotations

import numpy as np


def compare_predictions(previous: np.ndarray, current: np.ndarray) -> dict[str, float]:
    differences = np.abs(current - previous)
    return {
        "max_absolute_change": float(np.max(differences)),
        "mean_absolute_change": float(np.mean(differences)),
        "median_absolute_change": float(np.median(differences)),
    }


def convergence_difference(metrics: dict[str, float], metric: str) -> float:
    names = {
        "max": "max_absolute_change",
        "mean": "mean_absolute_change",
        "median": "median_absolute_change",
    }
    try:
        return metrics[names[metric]]
    except KeyError as exc:
        raise ValueError("convergence_metric must be 'max', 'mean', or 'median'") from exc
