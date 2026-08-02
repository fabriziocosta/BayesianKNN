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

from .model_families import make_model_estimator
from .models import ModelDraw
from .priors import GaussianCovarianceDraw, GaussianCovariancePrior, LogisticScalePrior
from .representation import make_representation
from .scoring import classification_cv_score, regression_cv_score

REPRESENTATION_FAMILIES = ("gaussian", "sparse", "identity")
MODEL_FAMILIES = ("knn", "linear", "gaussian")


@dataclass(frozen=True)
class SubsetSample:
    indices: np.ndarray
    log_probability: float


@dataclass(frozen=True)
class PreparedModel:
    """A sampled model before scoring and final fitting."""

    task: str
    model_family: str
    model_family_probability: float
    representation_family: str
    projection_dimension: int
    projection_parameters: dict[str, Any]
    representation_object: Any
    representation_family_probability: float
    gaussian_covariance_structure: str | None
    gaussian_covariance_probability: float
    gaussian_covariance_draw: GaussianCovarianceDraw | None
    subset_size: int
    subset_indices: np.ndarray
    subset_log_probability: float
    neighborhood_size: int
    projection_scale_draw: Any
    subset_scale_draw: Any
    neighbor_scale_draw: Any
    X_subset: Any
    y_subset: np.ndarray
    splits: list[tuple[np.ndarray, np.ndarray]]
    weights: str
    metric: str
    alpha: float
    epsilon: float
    classes: np.ndarray | None


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


def prepare_model(
    X: Any,
    y: np.ndarray,
    *,
    task: str,
    model_family: str,
    gaussian_covariance_prior: GaussianCovariancePrior,
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
) -> PreparedModel:
    rng = np.random.default_rng(seed)
    n_features = int(X.shape[1])
    if model_family == "mixed":
        selected_model_family = str(rng.choice(MODEL_FAMILIES))
        model_family_probability = 1.0 / len(MODEL_FAMILIES)
    elif model_family in MODEL_FAMILIES:
        selected_model_family = model_family
        model_family_probability = 1.0
    else:
        choices = ", ".join((*MODEL_FAMILIES, "mixed"))
        raise ValueError(f"model_family must be one of: {choices}")
    if selected_model_family == "knn" and representation == "mixed":
        representation_family = str(rng.choice(REPRESENTATION_FAMILIES))
        representation_family_probability = 1.0 / len(REPRESENTATION_FAMILIES)
    elif selected_model_family == "knn" and representation in REPRESENTATION_FAMILIES:
        representation_family = representation
        representation_family_probability = 1.0
    else:
        representation_family = "identity"
        representation_family_probability = 1.0
    projection_values = (
        [n_features]
        if representation_family == "identity"
        else range(1, n_features + 1)
    )
    projection_draw = scale_prior.draw(projection_values, rng)
    projection = make_representation(representation_family, projection_draw.value, seed)
    sampled_projection_parameters = projection.sample_parameters(seed)
    transformed = projection.fit_transform(X)

    if selected_model_family == "gaussian":
        covariance_draw = gaussian_covariance_prior.draw(n_features, rng)
        covariance_structure = covariance_draw.value
        covariance_probability = covariance_draw.probability
    else:
        covariance_draw = None
        covariance_structure = None
        covariance_probability = 1.0

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
    if selected_model_family == "knn":
        neighbor_draw = scale_prior.draw(range(1, k_max + 1), rng)
        neighborhood_size = neighbor_draw.value
    else:
        neighbor_draw = None
        neighborhood_size = None

    return PreparedModel(
        task=task,
        model_family=selected_model_family,
        model_family_probability=model_family_probability,
        representation_family=representation_family,
        projection_dimension=projection_draw.value,
        projection_parameters={**sampled_projection_parameters, **projection.parameters()},
        representation_object=projection,
        representation_family_probability=representation_family_probability,
        gaussian_covariance_structure=covariance_structure,
        gaussian_covariance_probability=covariance_probability,
        gaussian_covariance_draw=covariance_draw,
        subset_size=subset_draw.value,
        subset_indices=subset.indices,
        subset_log_probability=subset.log_probability,
        neighborhood_size=neighborhood_size,
        projection_scale_draw=projection_draw,
        subset_scale_draw=subset_draw,
        neighbor_scale_draw=neighbor_draw,
        X_subset=X_subset,
        y_subset=y_subset,
        splits=splits,
        weights=weights,
        metric=metric,
        alpha=alpha,
        epsilon=epsilon,
        classes=classes,
    )


def score_prepared_model(prepared: PreparedModel) -> float:
    """Score one prepared model; this phase is independently parallelizable."""

    if prepared.model_family == "knn" and prepared.task == "classification":
        return classification_cv_score(
            prepared.X_subset,
            prepared.y_subset,
            prepared.splits,
            prepared.neighborhood_size,
            prepared.weights,
            prepared.metric,
            prepared.alpha,
            prepared.classes,
        )
    if prepared.model_family == "knn":
        return regression_cv_score(
            prepared.X_subset,
            prepared.y_subset,
            prepared.splits,
            prepared.neighborhood_size,
            prepared.weights,
            prepared.metric,
            prepared.epsilon,
        )

    scores: list[float] = []
    class_positions = (
        {label: index for index, label in enumerate(prepared.classes)}
        if prepared.task == "classification"
        else None
    )
    for train_indices, validation_indices in prepared.splits:
        estimator = make_model_estimator(
            prepared.task,
            prepared.model_family,
            None,
            prepared.weights,
            prepared.metric,
            prepared.gaussian_covariance_structure,
        )
        estimator.fit(prepared.X_subset[train_indices], prepared.y_subset[train_indices])
        if prepared.task == "classification":
            probabilities = estimator.predict_proba(prepared.X_subset[validation_indices])
            aligned = np.zeros(
                (len(probabilities), len(prepared.classes)),
                dtype=float,
            )
            positions = {
                label: index for index, label in enumerate(estimator.classes_)
            }
            for local_index, label in enumerate(estimator.classes_):
                aligned[:, class_positions[label]] = probabilities[:, local_index]
            smoothed = (aligned + prepared.alpha) / (
                1.0 + len(prepared.classes) * prepared.alpha
            )
            target_positions = np.array(
                [class_positions[label] for label in prepared.y_subset[validation_indices]]
            )
            scores.extend(
                np.log(smoothed[np.arange(len(target_positions)), target_positions]).tolist()
            )
        else:
            predictions = estimator.predict(prepared.X_subset[validation_indices])
            sigma2 = max(float(np.var(prepared.y_subset[train_indices])), prepared.epsilon**2)
            scores.extend(
                (
                    -0.5
                    * (
                        np.log(2.0 * np.pi * sigma2)
                        + (prepared.y_subset[validation_indices] - predictions) ** 2 / sigma2
                    )
                ).tolist()
            )
    return float(np.mean(scores))


def fit_prepared_model(prepared: PreparedModel, cv_score: float) -> ModelDraw:
    """Fit one final model after its CV score has been computed."""

    estimator = make_model_estimator(
        prepared.task,
        prepared.model_family,
        prepared.neighborhood_size,
        prepared.weights,
        prepared.metric,
        prepared.gaussian_covariance_structure,
    )
    estimator.fit(prepared.X_subset, prepared.y_subset)
    log_prior = (
        np.log(prepared.model_family_probability)
        + np.log(prepared.representation_family_probability)
        + np.log(prepared.gaussian_covariance_probability)
        + prepared.projection_scale_draw.log_probability
        + prepared.subset_scale_draw.log_probability
        + prepared.subset_log_probability
    )
    if prepared.neighbor_scale_draw is not None:
        log_prior += prepared.neighbor_scale_draw.log_probability
    return ModelDraw(
        model_family=prepared.model_family,
        representation_family=prepared.representation_family,
        projection_dimension=prepared.projection_dimension,
        projection_parameters=prepared.projection_parameters,
        representation_object=prepared.representation_object,
        model_family_probability=prepared.model_family_probability,
        representation_family_probability=prepared.representation_family_probability,
        gaussian_covariance_structure=prepared.gaussian_covariance_structure,
        gaussian_covariance_probability=prepared.gaussian_covariance_probability,
        gaussian_covariance_draw=prepared.gaussian_covariance_draw,
        subset_size=prepared.subset_size,
        subset_indices=prepared.subset_indices,
        neighborhood_size=prepared.neighborhood_size,
        projection_scale_draw=prepared.projection_scale_draw,
        subset_scale_draw=prepared.subset_scale_draw,
        neighbor_scale_draw=prepared.neighbor_scale_draw,
        log_prior=float(log_prior),
        log_proposal=float(log_prior),
        cv_log_pseudo_likelihood=float(cv_score),
        estimator=estimator,
    )


def build_model(
    X: Any,
    y: np.ndarray,
    **kwargs: Any,
) -> ModelDraw:
    """Build one model sequentially for callers that do not need staged parallelism."""

    prepared = prepare_model(X, y, **kwargs)
    return fit_prepared_model(prepared, score_prepared_model(prepared))
