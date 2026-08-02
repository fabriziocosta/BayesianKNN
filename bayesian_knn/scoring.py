"""Cross-validated pseudo-likelihood scoring."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


def _aligned_probabilities(
    probabilities: np.ndarray, local_classes: np.ndarray, global_classes: np.ndarray
) -> np.ndarray:
    aligned = np.zeros((len(probabilities), len(global_classes)), dtype=float)
    positions = {label: index for index, label in enumerate(global_classes)}
    for local_index, label in enumerate(local_classes):
        aligned[:, positions[label]] = probabilities[:, local_index]
    return aligned


def classification_cv_score(
    X: Any,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    n_neighbors: int,
    weights: str,
    metric: str,
    alpha: float,
    classes: np.ndarray,
) -> float:
    class_positions = {label: index for index, label in enumerate(classes)}
    log_likelihoods: list[float] = []
    for train_indices, validation_indices in splits:
        estimator = KNeighborsClassifier(
            n_neighbors=n_neighbors, weights=weights, metric=metric, n_jobs=1
        )
        estimator.fit(X[train_indices], y[train_indices])
        probabilities = _aligned_probabilities(
            estimator.predict_proba(X[validation_indices]), estimator.classes_, classes
        )
        smoothed = (n_neighbors * probabilities + alpha) / (
            n_neighbors + len(classes) * alpha
        )
        target_positions = np.array([class_positions[label] for label in y[validation_indices]])
        log_likelihoods.extend(np.log(smoothed[np.arange(len(target_positions)), target_positions]))
    return float(np.mean(log_likelihoods))


def regression_cv_score(
    X: Any,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    n_neighbors: int,
    weights: str,
    metric: str,
    epsilon: float,
) -> float:
    log_likelihoods: list[float] = []
    for train_indices, validation_indices in splits:
        estimator = KNeighborsRegressor(
            n_neighbors=n_neighbors, weights=weights, metric=metric, n_jobs=1
        )
        estimator.fit(X[train_indices], y[train_indices])
        predictions = estimator.predict(X[validation_indices])
        sigma2 = max(float(np.var(y[train_indices])), epsilon**2)
        log_likelihoods.extend(
            -0.5
            * (
                np.log(2.0 * np.pi * sigma2)
                + (y[validation_indices] - predictions) ** 2 / sigma2
            )
        )
    return float(np.mean(log_likelihoods))
