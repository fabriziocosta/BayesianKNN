"""Generic cross-validated pseudo-likelihood scoring."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from .adapters import EstimatorFamilyAdapter
from .utils import child_seed


def fit_adapter_estimator(
    adapter: EstimatorFamilyAdapter,
    estimator: Any,
    X: Any,
    y: np.ndarray,
) -> None:
    """Fit an adapter estimator while suppressing expected MLP convergence noise."""

    with warnings.catch_warnings():
        if adapter.name == "mlp":
            warnings.filterwarnings(
                "ignore",
                message="Stochastic Optimizer: Maximum iterations.*",
                category=ConvergenceWarning,
            )
        estimator.fit(X, y)


def _aligned_probabilities(
    probabilities: np.ndarray, local_classes: np.ndarray, global_classes: np.ndarray
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(local_classes):
        raise ValueError("classifier probabilities and classes have incompatible shapes")
    if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError("classifier probabilities must be finite and non-negative")
    aligned = np.zeros((probabilities.shape[0], len(global_classes)), dtype=float)
    positions = {label: index for index, label in enumerate(global_classes)}
    for local_index, label in enumerate(local_classes):
        if label not in positions:
            raise ValueError("classifier returned a class not present in the global labels")
        aligned[:, positions[label]] = probabilities[:, local_index]
    row_sums = aligned.sum(axis=1)
    if np.any(row_sums <= 0):
        raise ValueError("classifier probabilities must have positive row sums")
    return aligned / row_sums[:, None]


def classification_cv_score(
    X: Any,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    adapter: EstimatorFamilyAdapter,
    parameters: Mapping[str, Any],
    alpha: float,
    classes: np.ndarray,
    seed: int,
) -> float:
    """Score any adapter that exposes a native ``predict_proba`` method."""

    class_positions = {label: index for index, label in enumerate(classes)}
    log_likelihoods: list[float] = []
    concentration = float(adapter.predictive_concentration("classification", parameters))
    if not np.isfinite(concentration) or concentration <= 0:
        raise ValueError("predictive concentration must be finite and positive")
    for fold_index, (train_indices, validation_indices) in enumerate(splits):
        estimator = adapter.build_estimator(
            "classification", parameters, child_seed(seed, fold_index)
        )
        fit_adapter_estimator(
            adapter,
            estimator,
            X[train_indices],
            y[train_indices],
        )
        if not callable(getattr(estimator, "predict_proba", None)):
            raise TypeError(
                f"classification adapter {adapter.name!r} must build an estimator with "
                "predict_proba"
            )
        probabilities = _aligned_probabilities(
            estimator.predict_proba(X[validation_indices]), estimator.classes_, classes
        )
        smoothed = (concentration * probabilities + alpha) / (
            concentration + len(classes) * alpha
        )
        target_positions = np.array([class_positions[label] for label in y[validation_indices]])
        log_likelihoods.extend(
            np.log(smoothed[np.arange(len(target_positions)), target_positions]).tolist()
        )
    if not log_likelihoods:
        raise ValueError("cross-validation produced no validation observations")
    return float(np.mean(log_likelihoods))


def regression_cv_score(
    X: Any,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    adapter: EstimatorFamilyAdapter,
    parameters: Mapping[str, Any],
    epsilon: float,
    seed: int,
) -> float:
    """Score any adapter that exposes a regression ``predict`` method."""

    log_likelihoods: list[float] = []
    for fold_index, (train_indices, validation_indices) in enumerate(splits):
        estimator = adapter.build_estimator("regression", parameters, child_seed(seed, fold_index))
        fit_adapter_estimator(
            adapter,
            estimator,
            X[train_indices],
            y[train_indices],
        )
        predictions = np.asarray(estimator.predict(X[validation_indices]), dtype=float)
        sigma2 = max(float(np.var(y[train_indices])), epsilon**2)
        log_likelihoods.extend(
            (
                -0.5
                * (
                    np.log(2.0 * np.pi * sigma2)
                    + (y[validation_indices] - predictions) ** 2 / sigma2
                )
            ).tolist()
        )
    if not log_likelihoods:
        raise ValueError("cross-validation produced no validation observations")
    return float(np.mean(log_likelihoods))
