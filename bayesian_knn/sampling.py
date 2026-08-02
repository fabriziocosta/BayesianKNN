"""Sampling of complete Bayesian k-NN model configurations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import gammaln, logsumexp
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from .models import ModelDraw
from .priors import LogisticScalePrior
from .representation import make_representation
from .scoring import classification_cv_score, regression_cv_score


@dataclass(frozen=True)
class SubsetSample:
    indices: np.ndarray
    log_probability: float


def n_splits_for_cv(cv: int | Any) -> int:
    if isinstance(cv, (int, np.integer)):
        if int(cv) < 2:
            raise ValueError("cv must be at least 2")
        return int(cv)
    try:
        value = int(cv.get_n_splits())
    except AttributeError as exc:
        raise ValueError("cv must be an integer or a scikit-learn splitter") from exc
    if value < 2:
        raise ValueError("cv must have at least two splits")
    return value


def make_cv_splitter(task: str, cv: int | Any, seed: int) -> Any:
    if isinstance(cv, (int, np.integer)):
        if task == "classification":
            return StratifiedKFold(n_splits=int(cv), shuffle=True, random_state=seed)
        return KFold(n_splits=int(cv), shuffle=True, random_state=seed)
    return clone(cv)


def feasible_subset_sizes(
    y: np.ndarray,
    task: str,
    minimum: int,
    maximum: int,
    n_splits: int,
) -> list[int]:
    if minimum < 1 or maximum < minimum:
        raise ValueError("subset bounds are invalid")
    if maximum > len(y):
        raise ValueError("max_subset_size cannot exceed n_samples")
    lower = n_splits if task == "regression" else n_splits * np.unique(y).size
    return [size for size in range(minimum, maximum + 1) if size >= lower]


def _log_combination(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return float(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1))


def sample_subset(
    y: np.ndarray,
    size: int,
    task: str,
    n_splits: int,
    rng: np.random.Generator,
) -> SubsetSample:
    """Sample uniformly from all CV-admissible subsets of a fixed size."""

    n_samples = len(y)
    if task == "regression":
        indices = np.sort(rng.choice(n_samples, size=size, replace=False)).astype(int)
        return SubsetSample(indices=indices, log_probability=-_log_combination(n_samples, size))

    labels, inverse = np.unique(y, return_inverse=True)
    class_indices = [np.flatnonzero(inverse == class_index) for class_index in range(len(labels))]
    class_counts = [len(indices) for indices in class_indices]
    counts, log_total = _sample_class_counts(class_counts, n_splits, size, rng)
    selected = [
        rng.choice(indices, size=int(count), replace=False)
        for indices, count in zip(class_indices, counts)
    ]
    return SubsetSample(
        indices=np.sort(np.concatenate(selected)).astype(int), log_probability=-log_total
    )


def _sample_class_counts(
    class_counts: Sequence[int], minimum_count: int, target_size: int, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    n_classes = len(class_counts)
    suffix = np.full((n_classes + 1, target_size + 1), -np.inf, dtype=float)
    suffix[n_classes, 0] = 0.0
    for class_index in range(n_classes - 1, -1, -1):
        n_items = int(class_counts[class_index])
        for total in range(target_size + 1):
            terms = [
                _log_combination(n_items, count) + suffix[class_index + 1, total - count]
                for count in range(minimum_count, min(n_items, total) + 1)
                if np.isfinite(suffix[class_index + 1, total - count])
            ]
            if terms:
                suffix[class_index, total] = logsumexp(terms)
    total_log_count = float(suffix[0, target_size])
    if not np.isfinite(total_log_count):
        raise ValueError("no admissible classification subset exists")

    counts = np.zeros(n_classes, dtype=int)
    remaining = target_size
    for class_index, n_items in enumerate(class_counts):
        possible = []
        terms = []
        for count in range(minimum_count, min(n_items, remaining) + 1):
            tail = suffix[class_index + 1, remaining - count]
            if np.isfinite(tail):
                possible.append(count)
                terms.append(_log_combination(n_items, count) + tail)
        probabilities = np.exp(np.asarray(terms) - logsumexp(terms))
        chosen = int(rng.choice(possible, p=probabilities))
        counts[class_index] = chosen
        remaining -= chosen
    return counts, total_log_count


def build_model(
    X: Any,
    y: np.ndarray,
    *,
    task: str,
    representation: str,
    scale_prior: LogisticScalePrior,
    min_subset_size: int,
    max_subset_size: int,
    max_neighbors: int | None,
    weights: str,
    metric: str,
    cv: int | Any,
    alpha: float,
    epsilon: float,
    seed: int,
    classes: np.ndarray | None,
) -> ModelDraw:
    rng = np.random.default_rng(seed)
    n_features = int(X.shape[1])
    projection_values = [n_features] if representation == "identity" else range(1, n_features + 1)
    projection_draw = scale_prior.draw(projection_values, rng)
    projection = make_representation(representation, projection_draw.value, seed)
    transformed = projection.fit_transform(X)

    n_splits = n_splits_for_cv(cv)
    feasible_sizes = feasible_subset_sizes(y, task, min_subset_size, max_subset_size, n_splits)
    if not feasible_sizes:
        raise ValueError("no CV-admissible subset size exists")
    subset_draw = scale_prior.draw(feasible_sizes, rng)
    subset = sample_subset(y, subset_draw.value, task, n_splits, rng)
    X_subset = transformed[subset.indices]
    y_subset = y[subset.indices]

    cv_splitter = make_cv_splitter(task, cv, int(rng.integers(0, 2**32 - 1)))
    splits = list(cv_splitter.split(X_subset, y_subset))
    n_train_min = min(len(train_indices) for train_indices, _ in splits)
    k_max = n_train_min if max_neighbors is None else min(max_neighbors, n_train_min)
    if k_max < 1:
        raise ValueError("max_neighbors leaves no valid neighbourhood size")
    neighbor_draw = scale_prior.draw(range(1, k_max + 1), rng)

    if task == "classification":
        cv_score = classification_cv_score(
            X_subset, y_subset, splits, neighbor_draw.value, weights, metric, alpha, classes
        )
        estimator = KNeighborsClassifier(
            n_neighbors=neighbor_draw.value, weights=weights, metric=metric, n_jobs=1
        )
    else:
        cv_score = regression_cv_score(
            X_subset, y_subset, splits, neighbor_draw.value, weights, metric, epsilon
        )
        estimator = KNeighborsRegressor(
            n_neighbors=neighbor_draw.value, weights=weights, metric=metric, n_jobs=1
        )
    estimator.fit(X_subset, y_subset)
    log_prior = (
        projection_draw.log_probability
        + subset_draw.log_probability
        + neighbor_draw.log_probability
        + subset.log_probability
    )
    return ModelDraw(
        representation_family=representation,
        projection_dimension=projection_draw.value,
        projection_parameters=projection.parameters(),
        representation_object=projection,
        subset_size=subset_draw.value,
        subset_indices=subset.indices,
        neighborhood_size=neighbor_draw.value,
        projection_scale_draw=projection_draw,
        subset_scale_draw=subset_draw,
        neighbor_scale_draw=neighbor_draw,
        log_prior=float(log_prior),
        log_proposal=float(log_prior),
        cv_log_pseudo_likelihood=float(cv_score),
        estimator=estimator,
    )
