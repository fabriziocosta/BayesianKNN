"""Shared sklearn estimator implementation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted

from .adapters import (
    EstimatorFamilyAdapter,
    FamilyRegistration,
    normalize_family_registry,
)
from .convergence import compare_predictions, convergence_difference
from .models import ModelDraw, aggregate_model_masses, recompute_importance_weights
from .priors import LogisticScalePrior, make_scale_prior
from .sampling import (
    feasible_subset_sizes,
    fit_prepared_model,
    n_splits_for_cv,
    prepare_model,
    score_prepared_model,
)
from .utils import base_seed, child_seed, stable_softmax


def _predict_single(model: ModelDraw, X: Any, task: str, classes: np.ndarray | None) -> np.ndarray:
    if task == "classification":
        probabilities = model.estimator.predict_proba(X)
        aligned = np.zeros((len(probabilities), len(classes)), dtype=float)
        positions = {label: index for index, label in enumerate(classes)}
        for local_index, label in enumerate(model.estimator.classes_):
            aligned[:, positions[label]] = probabilities[:, local_index]
        return aligned
    return np.asarray(model.estimator.predict(X), dtype=float)


class BayesianPredictiveModelAveragingBase(BaseEstimator):
    """Common Monte Carlo fitting and prediction machinery."""

    def __init__(
        self,
        family_registry: Sequence[FamilyRegistration | EstimatorFamilyAdapter] | None = None,
        scale_prior: LogisticScalePrior | None = None,
        min_subset_size: int | None = None,
        max_subset_size: int | None = None,
        cv: int | Any = 5,
        n_estimators: int | str = "auto",
        max_estimators: int = 1280,
        tolerance: float = 1e-3,
        convergence_metric: str = "max",
        convergence_size: int = 100,
        alpha: float = 1.0,
        epsilon: float = 1e-8,
        temperature: float = 1.0,
        n_jobs: int | None = -1,
        random_state: Any = None,
        adaptive_importance_sampling: bool = False,
        round_size: int = 50,
        min_rounds: int = 2,
        max_rounds: int | None = None,
        defensive_prior_weight: float = 0.2,
        proposal_tolerance: float = 1e-3,
        prediction_tolerance: float | None = None,
        ess_target_fraction: float | None = None,
        stopping_patience: int = 2,
        adaptation_temperature: float = 1.0,
    ) -> None:
        self.family_registry = family_registry
        self.scale_prior = scale_prior
        self.min_subset_size = min_subset_size
        self.max_subset_size = max_subset_size
        self.cv = cv
        self.n_estimators = n_estimators
        self.max_estimators = max_estimators
        self.tolerance = tolerance
        self.convergence_metric = convergence_metric
        self.convergence_size = convergence_size
        self.alpha = alpha
        self.epsilon = epsilon
        self.temperature = temperature
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.adaptive_importance_sampling = adaptive_importance_sampling
        self.round_size = round_size
        self.min_rounds = min_rounds
        self.max_rounds = max_rounds
        self.defensive_prior_weight = defensive_prior_weight
        self.proposal_tolerance = proposal_tolerance
        self.prediction_tolerance = prediction_tolerance
        self.ess_target_fraction = ess_target_fraction
        self.stopping_patience = stopping_patience
        self.adaptation_temperature = adaptation_temperature

    def _validate_parameters(self) -> None:
        if not isinstance(self.adaptive_importance_sampling, (bool, np.bool_)):
            raise ValueError("adaptive_importance_sampling must be boolean")
        if self.n_estimators != "auto" and (
            isinstance(self.n_estimators, bool)
            or not isinstance(self.n_estimators, (int, np.integer))
            or int(self.n_estimators) < 1
        ):
            raise ValueError("n_estimators must be a positive integer or 'auto'")
        if (
            isinstance(self.max_estimators, bool)
            or not isinstance(self.max_estimators, (int, np.integer))
            or int(self.max_estimators) < 1
        ):
            raise ValueError("max_estimators must be a positive integer")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        if self.convergence_metric not in {"max", "mean", "median"}:
            raise ValueError("convergence_metric must be 'max', 'mean', or 'median'")
        if isinstance(self.convergence_size, bool) or int(self.convergence_size) < 1:
            raise ValueError("convergence_size must be positive")
        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        for name, value in (
            ("round_size", self.round_size),
            ("min_rounds", self.min_rounds),
            ("stopping_patience", self.stopping_patience),
        ):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be a positive integer")
            if int(value) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_rounds is not None and (
            isinstance(self.max_rounds, (bool, np.bool_))
            or not isinstance(self.max_rounds, (int, np.integer))
            or int(self.max_rounds) < 1
        ):
            raise ValueError("max_rounds must be a positive integer or None")
        if not np.isfinite(self.defensive_prior_weight) or not (
            0 < self.defensive_prior_weight <= 1
        ):
            raise ValueError("defensive_prior_weight must be in (0, 1]")
        if not np.isfinite(self.proposal_tolerance) or self.proposal_tolerance <= 0:
            raise ValueError("proposal_tolerance must be finite and positive")
        if self.prediction_tolerance is not None and (
            not np.isfinite(self.prediction_tolerance) or self.prediction_tolerance <= 0
        ):
            raise ValueError("prediction_tolerance must be finite and positive or None")
        if self.ess_target_fraction is not None and (
            not np.isfinite(self.ess_target_fraction)
            or not 0 < self.ess_target_fraction <= 1
        ):
            raise ValueError("ess_target_fraction must be in (0, 1] or None")
        if not np.isfinite(self.adaptation_temperature) or self.adaptation_temperature <= 0:
            raise ValueError("adaptation_temperature must be finite and positive")
        for name, value in (
            ("min_subset_size", self.min_subset_size),
            ("max_subset_size", self.max_subset_size),
        ):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or int(value) < 1
            ):
                raise ValueError(f"{name} must be a positive integer")

    def _fit_task(self, X: Any, y: Any, task: str) -> BayesianPredictiveModelAveragingBase:
        self._validate_parameters()
        X, y = check_X_y(X, y, accept_sparse=True, ensure_2d=True, y_numeric=task == "regression")
        if task == "regression":
            y = np.asarray(y, dtype=float)
            self.classes_ = None
        else:
            self.classes_ = np.unique(y)
            if len(self.classes_) < 2:
                raise ValueError("classification requires at least two classes")
        self.n_features_in_ = int(X.shape[1])
        n_splits = n_splits_for_cv(self.cv)
        if task == "classification" and np.any(
            np.bincount(np.searchsorted(self.classes_, y), minlength=len(self.classes_)) < n_splits
        ):
            raise ValueError("each class must contain at least cv observations")

        minimum = int(self.min_subset_size) if self.min_subset_size is not None else (
            n_splits if task == "regression" else n_splits * len(self.classes_)
        )
        maximum = int(self.max_subset_size) if self.max_subset_size is not None else len(y)
        feasible = feasible_subset_sizes(y, task, minimum, maximum, n_splits)
        if not feasible:
            raise ValueError("no admissible subset size exists for this dataset and CV setup")

        self.scale_prior_ = make_scale_prior(self.scale_prior)
        registry = normalize_family_registry(self.family_registry)
        compatible = tuple(
            registration
            for registration in registry
            if task in registration.adapter.supported_tasks
        )
        if not compatible:
            raise ValueError(f"no registered adapter supports task {task!r}")
        compatible_weight = sum(registration.prior_weight for registration in compatible)
        self.family_registry_ = tuple(
            FamilyRegistration(
                registration.adapter,
                registration.prior_weight / compatible_weight,
            )
            for registration in compatible
        )
        self._base_seed = base_seed(self.random_state)
        self._task = task
        self._X_fit_shape = X.shape
        convergence_rng = np.random.default_rng(child_seed(self._base_seed, -1))
        n_samples = int(X.shape[0])
        convergence_size = min(int(self.convergence_size), n_samples)
        self.convergence_subset_indices_ = np.sort(
            convergence_rng.choice(n_samples, size=convergence_size, replace=False)
        )
        self.convergence_history_ = []

        self.proposal_history_: list[dict[str, float]] = []
        self.round_history_: list[dict[str, Any]] = []
        self.n_rounds_ = 0
        self.effective_sample_size_ = None
        self.effective_sample_size_fraction_ = None
        self.adaptive_converged_ = False
        self.stopping_reason_ = None
        self._round_sizes_: list[int] = []
        self._prior_family_probabilities_ = {
            registration.adapter.name: float(registration.prior_weight)
            for registration in self.family_registry_
        }

        if self.n_estimators == "auto":
            target = min(20, int(self.max_estimators))
            auto = True
        else:
            target = int(self.n_estimators)
            auto = False
        previous_prediction = None
        self._models: list[ModelDraw] = []
        self.converged_ = None if not auto else False

        if self.adaptive_importance_sampling:
            self._fit_adaptive(X, y)
            self.n_estimators_ = len(self._models)
            self.converged_ = self.adaptive_converged_
            return self

        while True:
            start = len(self._models)
            seeds = [child_seed(self._base_seed, index) for index in range(start, target)]
            prepared_models = Parallel(n_jobs=self.n_jobs)(
                delayed(self._prepare_model)(X, y, seed) for seed in seeds
            )
            scores = Parallel(n_jobs=self.n_jobs)(
                delayed(score_prepared_model)(prepared) for prepared in prepared_models
            )
            new_models = Parallel(n_jobs=self.n_jobs)(
                delayed(fit_prepared_model)(prepared, score)
                for prepared, score in zip(prepared_models, scores)
            )
            self._models.extend(new_models)
            self._update_weights()

            if not auto:
                break
            current_prediction = self._predict_outputs(X[self.convergence_subset_indices_])
            if previous_prediction is not None:
                metrics = compare_predictions(previous_prediction, current_prediction)
                metrics["n_estimators"] = len(self._models)
                metrics["difference"] = convergence_difference(metrics, self.convergence_metric)
                self.convergence_history_.append(metrics)
                if metrics["difference"] <= self.tolerance:
                    self.converged_ = True
                    break
            previous_prediction = current_prediction
            if len(self._models) >= int(self.max_estimators):
                break
            target = min(len(self._models) * 2, int(self.max_estimators))

        self.n_estimators_ = len(self._models)
        self.effective_sample_size_, self.effective_sample_size_fraction_ = (
            recompute_importance_weights(
                self._models,
                target_temperature=self.temperature,
                adaptive=False,
            )
        )
        return self

    def _prepare_model(
        self,
        X: Any,
        y: np.ndarray,
        seed: int,
        family_proposal_probabilities: Sequence[float] | None = None,
        round_index: int = 0,
        proposal_id: str = "prior-0",
    ) -> Any:
        return prepare_model(
            X,
            y,
            task=self._task,
            family_registry=self.family_registry_,
            scale_prior=self.scale_prior_,
            min_subset_size=(
                int(self.min_subset_size)
                if self.min_subset_size is not None
                else (
                    n_splits_for_cv(self.cv)
                    if self._task == "regression"
                    else n_splits_for_cv(self.cv) * len(self.classes_)
                )
            ),
            max_subset_size=(
                int(self.max_subset_size) if self.max_subset_size is not None else len(y)
            ),
            cv=self.cv,
            alpha=self.alpha,
            epsilon=self.epsilon,
            seed=seed,
            classes=self.classes_,
            family_proposal_probabilities=family_proposal_probabilities,
            round_index=round_index,
            proposal_id=proposal_id,
        )

    def _fit_batch(
        self,
        X: Any,
        y: np.ndarray,
        *,
        start: int,
        count: int,
        family_proposal: dict[str, float] | None = None,
        round_index: int = 0,
        proposal_id: str = "prior-0",
    ) -> list[ModelDraw]:
        seeds = [child_seed(self._base_seed, index) for index in range(start, start + count)]
        proposal_values = None
        if family_proposal is not None:
            proposal_values = [
                family_proposal[registration.adapter.name]
                for registration in self.family_registry_
            ]
        prepared_models = Parallel(n_jobs=self.n_jobs)(
            delayed(self._prepare_model)(
                X,
                y,
                seed,
                proposal_values,
                round_index,
                proposal_id,
            )
            for seed in seeds
        )
        scores = Parallel(n_jobs=self.n_jobs)(
            delayed(score_prepared_model)(prepared) for prepared in prepared_models
        )
        return Parallel(n_jobs=self.n_jobs)(
            delayed(fit_prepared_model)(prepared, score)
            for prepared, score in zip(prepared_models, scores)
        )

    def _update_adaptive_weights(self) -> tuple[float, float]:
        ess, ess_fraction = recompute_importance_weights(
            self._models,
            target_temperature=self.temperature,
            adaptive=True,
            proposal_history=self.proposal_history_,
            round_sizes=self._round_sizes_,
        )
        self.model_masses_ = aggregate_model_masses(self._models)
        self.effective_sample_size_ = ess
        self.effective_sample_size_fraction_ = ess_fraction
        return ess, ess_fraction

    def _next_family_proposal(self) -> dict[str, float]:
        family_mass = self.model_masses_["family"]
        names = [registration.adapter.name for registration in self.family_registry_]
        weighted_mass = np.asarray([family_mass.get(name, 0.0) for name in names], dtype=float)
        weighted_mass = stable_softmax(np.log(np.maximum(weighted_mass, np.finfo(float).tiny)))
        adapted = stable_softmax(np.log(weighted_mass) / self.adaptation_temperature)
        proposal = (
            self.defensive_prior_weight
            * np.asarray([self._prior_family_probabilities_[name] for name in names])
            + (1.0 - self.defensive_prior_weight) * adapted
        )
        proposal /= proposal.sum()
        return {name: float(value) for name, value in zip(names, proposal)}

    def _fit_adaptive(self, X: Any, y: np.ndarray) -> None:
        proposal = dict(self._prior_family_probabilities_)
        previous_proposal: dict[str, float] | None = None
        previous_prediction = None
        stable_rounds = 0
        round_limit = self.max_rounds or int(np.ceil(self.max_estimators / self.round_size))

        for round_index in range(round_limit):
            remaining = int(self.max_estimators) - len(self._models)
            if remaining <= 0:
                self.stopping_reason_ = "max_estimators"
                break
            count = min(int(self.round_size), remaining)
            proposal_id = f"round-{round_index}"
            new_models = self._fit_batch(
                X,
                y,
                start=len(self._models),
                count=count,
                family_proposal=proposal,
                round_index=round_index,
                proposal_id=proposal_id,
            )
            self._models.extend(new_models)
            self.proposal_history_.append(dict(proposal))
            self._round_sizes_.append(count)
            ess, ess_fraction = self._update_adaptive_weights()

            proposal_distance = None
            if previous_proposal is not None:
                proposal_distance = float(
                    0.5
                    * sum(
                        abs(proposal[name] - previous_proposal[name])
                        for name in proposal
                    )
                )
            prediction_change = None
            current_prediction = None
            if self.prediction_tolerance is not None:
                current_prediction = self._predict_outputs(X[self.convergence_subset_indices_])
                if previous_prediction is not None:
                    metrics = compare_predictions(previous_prediction, current_prediction)
                    prediction_change = convergence_difference(metrics, self.convergence_metric)
                    metrics["n_estimators"] = len(self._models)
                    metrics["difference"] = prediction_change
                    self.convergence_history_.append(metrics)

            proposal_stable = (
                proposal_distance is not None
                and proposal_distance <= self.proposal_tolerance
            )
            prediction_stable = (
                self.prediction_tolerance is None
                or (
                    prediction_change is not None
                    and prediction_change <= self.prediction_tolerance
                )
            )
            ess_stable = (
                self.ess_target_fraction is None
                or ess_fraction >= self.ess_target_fraction
            )
            conditions_met = (
                round_index + 1 >= int(self.min_rounds)
                and proposal_stable
                and prediction_stable
                and ess_stable
            )
            stable_rounds = stable_rounds + 1 if conditions_met else 0
            converged = stable_rounds >= int(self.stopping_patience)
            stop_reason = "converged" if converged else None
            if not converged and len(self._models) >= int(self.max_estimators):
                stop_reason = "max_estimators"
            elif not converged and round_index + 1 >= round_limit:
                stop_reason = "max_rounds"

            self.round_history_.append(
                {
                    "round_index": round_index,
                    "new_draws": count,
                    "cumulative_draws": len(self._models),
                    "proposal_probabilities": dict(proposal),
                    "posterior_family_mass": dict(self.model_masses_["family"]),
                    "proposal_distance": proposal_distance,
                    "prediction_change": prediction_change,
                    "effective_sample_size": ess,
                    "effective_sample_size_fraction": ess_fraction,
                    "maximum_normalized_weight": max(
                        model.posterior_weight for model in self._models
                    ),
                    "convergence_conditions_met": conditions_met,
                    "stopping_reason": stop_reason,
                }
            )
            self.n_rounds_ = round_index + 1
            if converged:
                self.adaptive_converged_ = True
                self.stopping_reason_ = "converged"
                break
            if stop_reason is not None:
                self.stopping_reason_ = stop_reason
                break

            previous_proposal = proposal
            if current_prediction is not None:
                previous_prediction = current_prediction
            proposal = self._next_family_proposal()

    def _update_weights(self) -> None:
        recompute_importance_weights(
            self._models,
            target_temperature=self.temperature,
            adaptive=False,
        )
        self.model_masses_ = aggregate_model_masses(self._models)

    def _validate_predict_X(self, X: Any) -> Any:
        check_is_fitted(self, "_models")
        X = check_array(X, accept_sparse=True, ensure_2d=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the fitted data")
        return X

    def _predict_outputs(self, X: Any) -> np.ndarray:
        X = self._validate_predict_X(X)
        outputs = Parallel(n_jobs=self.n_jobs)(
            delayed(_predict_single)(model, X, self._task, self.classes_)
            for model in self._models
        )
        stacked = np.stack(outputs, axis=0)
        weights = np.array([model.posterior_weight for model in self._models])
        return np.tensordot(weights, stacked, axes=(0, 0))

    def _predict_proba(self, X: Any) -> np.ndarray:
        return self._predict_outputs(X)

    def _predict(self, X: Any) -> np.ndarray:
        outputs = self._predict_outputs(X)
        if self._task == "classification":
            return self.classes_[np.argmax(outputs, axis=1)]
        return outputs

    def get_model_draws(self) -> list[dict[str, Any]]:
        check_is_fitted(self, "_models")
        return [model.to_dict() for model in self._models]

    def get_model_masses(self) -> dict[str, Any]:
        """Return normalized predictive shares by registered family name."""

        check_is_fitted(self, "_models")
        return aggregate_model_masses(self._models)
