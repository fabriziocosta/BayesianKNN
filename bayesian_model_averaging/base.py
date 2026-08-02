"""Shared sklearn estimator implementation."""

from __future__ import annotations

from typing import Any

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted

from .convergence import compare_predictions, convergence_difference
from .models import ModelDraw, aggregate_model_masses
from .priors import (
    GaussianCovariancePrior,
    LogisticScalePrior,
    make_gaussian_covariance_prior,
    make_scale_prior,
)
from .sampling import (
    feasible_subset_sizes,
    fit_prepared_model,
    n_splits_for_cv,
    prepare_model,
    score_prepared_model,
)
from .utils import base_seed, child_seed, stable_softmax


def _predict_single(model: ModelDraw, X: Any, task: str, classes: np.ndarray | None) -> np.ndarray:
    transformed = model.representation_object.transform(X)
    if task == "classification":
        probabilities = model.estimator.predict_proba(transformed)
        aligned = np.zeros((len(probabilities), len(classes)), dtype=float)
        positions = {label: index for index, label in enumerate(classes)}
        for local_index, label in enumerate(model.estimator.classes_):
            aligned[:, positions[label]] = probabilities[:, local_index]
        return aligned
    return np.asarray(model.estimator.predict(transformed), dtype=float)


class BayesianModelAveragingBase(BaseEstimator):
    """Common Monte Carlo fitting and prediction machinery."""

    def __init__(
        self,
        representation: str = "mixed",
        model_family: str = "mixed",
        scale_prior: LogisticScalePrior | None = None,
        gaussian_covariance_prior: GaussianCovariancePrior | None = None,
        min_subset_size: int | None = None,
        max_subset_size: int | None = None,
        max_neighbors: int | None = None,
        weights: str = "distance",
        metric: str = "euclidean",
        cv: int | Any = 5,
        n_estimators: int | str = "auto",
        max_estimators: int = 1280,
        tolerance: float = 1e-3,
        convergence_metric: str = "max",
        convergence_size: int = 100,
        alpha: float = 1.0,
        epsilon: float = 1e-8,
        n_jobs: int | None = -1,
        random_state: Any = None,
    ) -> None:
        self.representation = representation
        self.model_family = model_family
        self.scale_prior = scale_prior
        self.gaussian_covariance_prior = gaussian_covariance_prior
        self.min_subset_size = min_subset_size
        self.max_subset_size = max_subset_size
        self.max_neighbors = max_neighbors
        self.weights = weights
        self.metric = metric
        self.cv = cv
        self.n_estimators = n_estimators
        self.max_estimators = max_estimators
        self.tolerance = tolerance
        self.convergence_metric = convergence_metric
        self.convergence_size = convergence_size
        self.alpha = alpha
        self.epsilon = epsilon
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _validate_parameters(self) -> None:
        if self.representation not in {"gaussian", "sparse", "identity", "mixed"}:
            raise ValueError(
                "representation must be 'mixed', 'gaussian', 'sparse', or 'identity'"
            )
        if self.model_family not in {"knn", "linear", "gaussian", "mixed"}:
            raise ValueError("model_family must be 'knn', 'linear', 'gaussian', or 'mixed'")
        if self.weights not in {"uniform", "distance"}:
            raise ValueError("weights must be 'uniform' or 'distance'")
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
        for name, value in (
            ("min_subset_size", self.min_subset_size),
            ("max_subset_size", self.max_subset_size),
            ("max_neighbors", self.max_neighbors),
        ):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or int(value) < 1
            ):
                raise ValueError(f"{name} must be a positive integer")

    def _fit_task(self, X: Any, y: Any, task: str) -> BayesianModelAveragingBase:
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
        self.gaussian_covariance_prior_ = make_gaussian_covariance_prior(
            self.gaussian_covariance_prior
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

        if self.n_estimators == "auto":
            target = min(20, int(self.max_estimators))
            auto = True
        else:
            target = int(self.n_estimators)
            auto = False
        previous_prediction = None
        self._models: list[ModelDraw] = []
        self.converged_ = None if not auto else False

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
        return self

    def _prepare_model(self, X: Any, y: np.ndarray, seed: int) -> Any:
        return prepare_model(
            X,
            y,
            task=self._task,
            model_family=self.model_family,
            gaussian_covariance_prior=self.gaussian_covariance_prior_,
            representation=self.representation,
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
            max_neighbors=self.max_neighbors,
            weights=self.weights,
            metric=self.metric,
            cv=self.cv,
            alpha=self.alpha,
            epsilon=self.epsilon,
            seed=seed,
            classes=self.classes_,
        )

    def _update_weights(self) -> None:
        weights = stable_softmax(np.array([model.log_importance_weight for model in self._models]))
        for model, weight in zip(self._models, weights):
            model.posterior_weight = float(weight)
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
        """Return posterior mass by model family and family-specific choice.

        Family-specific ``*_size`` and ``*_structure`` masses are joint masses
        on the full ensemble and therefore sum to their parent family mass.
        The corresponding ``*_conditional`` mappings normalize within that
        family and sum to one whenever the family has positive mass.
        """

        check_is_fitted(self, "_models")
        return aggregate_model_masses(self._models)
