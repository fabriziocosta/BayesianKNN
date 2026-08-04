"""Sampling of complete Bayesian model averaging model configurations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import issparse
from scipy.special import gammaln, logsumexp
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold

from .adapters import (
    EstimatorFamilyAdapter,
    FamilyRegistration,
    SamplingContext,
)
from .models import ModelDraw
from .priors import LogisticScalePrior, ParameterDraw
from .scoring import classification_cv_score, fit_adapter_estimator, regression_cv_score


@dataclass(frozen=True)
class SubsetSample:
    indices: np.ndarray
    log_probability: float


@dataclass(frozen=True)
class PreparedModel:
    """A sampled model before scoring and final fitting."""

    task: str
    family_name: str
    family_adapter: EstimatorFamilyAdapter
    family_prior_probability: float
    family_proposal_probability: float
    round_index: int
    proposal_id: str
    parameters: dict[str, Any]
    parameter_prior: ParameterDraw
    subset_size: int
    subset_indices: np.ndarray
    subset_log_probability: float
    subset_scale_draw: Any
    X_subset: Any
    y_subset: np.ndarray
    splits: list[tuple[np.ndarray, np.ndarray]]
    alpha: float
    epsilon: float
    classes: np.ndarray | None
    seed: int


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
    family_registry: Sequence[FamilyRegistration],
    scale_prior: LogisticScalePrior,
    min_subset_size: int,
    max_subset_size: int,
    cv: int | Any,
    alpha: float,
    epsilon: float,
    seed: int,
    classes: np.ndarray | None,
    family_proposal_probabilities: Sequence[float] | None = None,
    round_index: int = 0,
    proposal_id: str = "prior-0",
) -> PreparedModel:
    rng = np.random.default_rng(seed)
    prior_probabilities = np.asarray(
        [registration.prior_weight for registration in family_registry], dtype=float
    )
    proposal_probabilities = (
        prior_probabilities
        if family_proposal_probabilities is None
        else np.asarray(family_proposal_probabilities, dtype=float)
    )
    if proposal_probabilities.shape != prior_probabilities.shape:
        raise ValueError("family proposal probabilities must match the family registry")
    if (
        np.any(~np.isfinite(proposal_probabilities))
        or np.any(proposal_probabilities <= 0)
        or not np.isclose(proposal_probabilities.sum(), 1.0)
    ):
        raise ValueError("family proposal probabilities must be finite, positive, and normalized")
    family_index = int(
        rng.choice(
            len(family_registry),
            p=proposal_probabilities,
        )
    )
    registration = family_registry[family_index]
    family_adapter = registration.adapter
    family_name = family_adapter.name
    if task not in family_adapter.supported_tasks:
        raise ValueError(f"adapter {family_name!r} does not support task {task!r}")
    n_splits = n_splits_for_cv(cv)
    feasible_sizes = feasible_subset_sizes(y, task, min_subset_size, max_subset_size, n_splits)
    if not feasible_sizes:
        raise ValueError("no CV-admissible subset size exists")
    subset_draw = scale_prior.draw(feasible_sizes, rng)
    subset = sample_subset(y, subset_draw.value, task, n_splits, rng)
    X_subset = X[subset.indices]
    y_subset = y[subset.indices]

    cv_splitter = make_cv_splitter(task, cv, int(rng.integers(0, 2**32 - 1)))
    splits = list(cv_splitter.split(X_subset, y_subset))
    n_train_min = min(len(train_indices) for train_indices, _ in splits)
    min_class_train_size = None
    min_class_distinct_train_size = None
    if task == "classification":
        subset_classes = np.unique(y_subset)
        min_class_train_size = min(
            int(np.count_nonzero(y_subset[train_indices] == label))
            for train_indices, _ in splits
            for label in subset_classes
        )
        distinct_counts = []
        for train_indices, _ in splits:
            X_train = X_subset[train_indices]
            if issparse(X_train):
                X_train = X_train.toarray()
            for label in subset_classes:
                class_rows = np.asarray(X_train)[y_subset[train_indices] == label]
                distinct_counts.append(len(np.unique(class_rows, axis=0)))
        min_class_distinct_train_size = min(distinct_counts)
    context = SamplingContext(
        task=task,
        n_features=int(X.shape[1]),
        n_classes=None if classes is None else len(classes),
        n_samples=len(y_subset),
        subset_size=int(subset_draw.value),
        min_train_size=n_train_min,
        classes=classes,
        scale_prior=scale_prior,
        min_class_train_size=min_class_train_size,
        min_class_distinct_train_size=min_class_distinct_train_size,
    )
    parameter_prior = family_adapter.sample_parameters(context, rng)
    if not np.isfinite(parameter_prior.log_probability):
        raise ValueError(f"adapter {family_name!r} returned a non-finite parameter prior")

    return PreparedModel(
        task=task,
        family_name=family_name,
        family_adapter=family_adapter,
        family_prior_probability=registration.prior_weight,
        family_proposal_probability=float(proposal_probabilities[family_index]),
        round_index=int(round_index),
        proposal_id=str(proposal_id),
        parameters=dict(parameter_prior.parameters),
        parameter_prior=parameter_prior,
        subset_size=subset_draw.value,
        subset_indices=subset.indices,
        subset_log_probability=subset.log_probability,
        subset_scale_draw=subset_draw,
        X_subset=X_subset,
        y_subset=y_subset,
        splits=splits,
        alpha=alpha,
        epsilon=epsilon,
        classes=classes,
        seed=seed,
    )


def score_prepared_model(prepared: PreparedModel) -> float:
    """Score one prepared model; this phase is independently parallelizable."""

    if prepared.task == "classification":
        return classification_cv_score(
            prepared.X_subset,
            prepared.y_subset,
            prepared.splits,
            prepared.family_adapter,
            prepared.parameters,
            prepared.alpha,
            prepared.classes,
            prepared.seed,
        )
    return regression_cv_score(
        prepared.X_subset,
        prepared.y_subset,
        prepared.splits,
        prepared.family_adapter,
        prepared.parameters,
        prepared.epsilon,
        prepared.seed,
    )


def fit_prepared_model(prepared: PreparedModel, cv_score: float) -> ModelDraw:
    """Fit one final model after its CV score has been computed."""

    estimator = prepared.family_adapter.build_estimator(
        prepared.task,
        prepared.parameters,
        prepared.seed,
    )
    fit_adapter_estimator(
        prepared.family_adapter,
        estimator,
        prepared.X_subset,
        prepared.y_subset,
    )
    log_prior = (
        np.log(prepared.family_prior_probability)
        + prepared.subset_scale_draw.log_probability
        + prepared.subset_log_probability
        + prepared.parameter_prior.log_probability
    )
    log_proposal = (
        log_prior
        - np.log(prepared.family_prior_probability)
        + np.log(prepared.family_proposal_probability)
    )
    return ModelDraw(
        family_name=prepared.family_name,
        family_prior_probability=prepared.family_prior_probability,
        family_proposal_probability=prepared.family_proposal_probability,
        round_index=prepared.round_index,
        proposal_id=prepared.proposal_id,
        parameters=prepared.parameters,
        parameter_prior=prepared.parameter_prior,
        subset_size=prepared.subset_size,
        subset_indices=prepared.subset_indices,
        subset_scale_draw=prepared.subset_scale_draw,
        log_prior=float(log_prior),
        log_proposal=float(log_proposal),
        cv_log_pseudo_likelihood=float(cv_score),
        estimator=estimator,
        generating_log_proposal=float(log_proposal),
    )


def build_model(
    X: Any,
    y: np.ndarray,
    **kwargs: Any,
) -> ModelDraw:
    """Build one model sequentially for callers that do not need staged parallelism."""

    prepared = prepare_model(X, y, **kwargs)
    return fit_prepared_model(prepared, score_prepared_model(prepared))
