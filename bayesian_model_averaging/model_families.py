"""Predictive estimators used by the model-family ensemble."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import issparse
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.mixture import GaussianMixture


def _dense(X: Any) -> np.ndarray:
    if issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=float)


def _standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(X, axis=0)
    scale = np.std(X, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (X - mean) / scale, mean, scale


def _standardize_transform(X: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (X - mean) / scale


def _gate_probabilities(X: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(X)), X])
    logits = design @ coefficients.T
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def _fit_gate(
    X: np.ndarray,
    responsibilities: np.ndarray,
    alpha: float,
    initial_coefficients: np.ndarray,
    max_iter: int,
) -> np.ndarray:
    n_experts = responsibilities.shape[1]
    if n_experts == 1:
        return np.zeros((1, X.shape[1] + 1), dtype=float)
    design = np.column_stack([np.ones(len(X)), X])
    initial = initial_coefficients[:-1].ravel()

    def objective_gradient(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        coefficients = np.zeros((n_experts, design.shape[1]), dtype=float)
        coefficients[:-1] = parameters.reshape(n_experts - 1, -1)
        log_probabilities = design @ coefficients.T
        log_probabilities -= logsumexp(log_probabilities, axis=1, keepdims=True)
        probabilities = np.exp(log_probabilities)
        loss = -float(np.mean(np.sum(responsibilities * log_probabilities, axis=1)))
        loss += 0.5 * alpha * float(np.sum(coefficients[:-1] ** 2))
        gradient = (probabilities - responsibilities).T @ design / len(X)
        gradient[:-1] += alpha * coefficients[:-1]
        return loss, gradient[:-1].ravel()

    result = minimize(
        objective_gradient,
        initial,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )
    coefficients = np.zeros((n_experts, design.shape[1]), dtype=float)
    coefficients[:-1] = result.x.reshape(n_experts - 1, -1)
    return coefficients


class GaussianClassifier:
    """Generative Gaussian classifier with a selected covariance structure."""

    def __init__(self, covariance_structure: str, reg_covar: float = 1e-6) -> None:
        self.covariance_structure = covariance_structure
        self.reg_covar = float(reg_covar)

    def fit(self, X: Any, y: Any) -> GaussianClassifier:
        X = _dense(X)
        if self.covariance_structure not in {"isotropic", "diagonal", "full"}:
            raise ValueError("covariance_structure must be isotropic, diagonal, or full")
        self.classes_, counts = np.unique(y, return_counts=True)
        self.class_log_prior_ = np.log(counts / counts.sum())
        self.means_ = np.asarray([X[y == label].mean(axis=0) for label in self.classes_])
        self.precisions_ = []
        self.log_determinants_ = []
        for mean, label in zip(self.means_, self.classes_):
            residuals = X[y == label] - mean
            if self.covariance_structure == "isotropic":
                variance = float(np.mean(residuals**2)) + self.reg_covar
                covariance = np.eye(X.shape[1]) * variance
            elif self.covariance_structure == "diagonal":
                variance = np.var(residuals, axis=0) + self.reg_covar
                covariance = np.diag(variance)
            else:
                denominator = max(residuals.shape[0] - 1, 1)
                covariance = residuals.T @ residuals / denominator
                covariance = covariance + self.reg_covar * np.eye(X.shape[1])
            sign, log_determinant = np.linalg.slogdet(covariance)
            if sign <= 0:
                raise ValueError("Gaussian covariance estimate is not positive definite")
            self.precisions_.append(np.linalg.inv(covariance))
            self.log_determinants_.append(float(log_determinant))
        self.precisions_ = np.asarray(self.precisions_)
        self.log_determinants_ = np.asarray(self.log_determinants_)
        return self

    def predict_log_proba(self, X: Any) -> np.ndarray:
        X = _dense(X)
        residuals = X[:, None, :] - self.means_[None, :, :]
        mahalanobis = np.einsum(
            "ncp,cpq,ncq->nc", residuals, self.precisions_, residuals
        )
        log_scores = self.class_log_prior_ - 0.5 * (
            X.shape[1] * np.log(2.0 * np.pi) + self.log_determinants_ + mahalanobis
        )
        return log_scores - logsumexp(log_scores, axis=1, keepdims=True)

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.exp(self.predict_log_proba(X))

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]


class GaussianMixtureClassifier:
    """Generative classifier with one Gaussian mixture fitted per class."""

    _covariance_types = {
        "isotropic": "spherical",
        "diagonal": "diag",
        "full": "full",
    }

    def __init__(
        self,
        n_components: int,
        covariance_structure: str,
        reg_covar: float = 1e-6,
        max_iter: int = 100,
        random_state: int | None = None,
    ) -> None:
        self.n_components = int(n_components)
        self.covariance_structure = covariance_structure
        self.reg_covar = float(reg_covar)
        self.max_iter = int(max_iter)
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> GaussianMixtureClassifier:
        X = _dense(X)
        if self.n_components < 1:
            raise ValueError("n_components must be positive")
        try:
            covariance_type = self._covariance_types[self.covariance_structure]
        except KeyError as exc:
            raise ValueError(
                "covariance_structure must be isotropic, diagonal, or full"
            ) from exc
        self.classes_, counts = np.unique(y, return_counts=True)
        if np.any(counts < self.n_components):
            raise ValueError("n_components cannot exceed samples in every class")
        self.class_log_prior_ = np.log(counts / counts.sum())
        self.mixtures_ = []
        for class_index, label in enumerate(self.classes_):
            class_samples = X[y == label]
            if len(class_samples) == 1:
                # GaussianMixture requires two rows even for one component.
                # The regularization then supplies the covariance for this
                # degenerate CV fold without changing the class location.
                class_samples = np.repeat(class_samples, 2, axis=0)
            mixture = GaussianMixture(
                n_components=self.n_components,
                covariance_type=covariance_type,
                reg_covar=self.reg_covar,
                max_iter=self.max_iter,
                n_init=1,
                random_state=(
                    None
                    if self.random_state is None
                    else int(self.random_state) + class_index
                ),
            )
            mixture.fit(class_samples)
            self.mixtures_.append(mixture)
        return self

    def predict_log_proba(self, X: Any) -> np.ndarray:
        X = _dense(X)
        log_scores = np.column_stack(
            [mixture.score_samples(X) for mixture in self.mixtures_]
        )
        log_scores += self.class_log_prior_[None, :]
        return log_scores - logsumexp(log_scores, axis=1, keepdims=True)

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.exp(self.predict_log_proba(X))

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]


class GatedLinearExpertsClassifier:
    """Mixture of logistic experts with a learned linear softmax gate."""

    def __init__(
        self,
        n_experts: int,
        expert_alpha: float,
        gating_alpha: float,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: int | None = None,
    ) -> None:
        self.n_experts = int(n_experts)
        self.expert_alpha = float(expert_alpha)
        self.gating_alpha = float(gating_alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> GatedLinearExpertsClassifier:
        X = _dense(X)
        if self.n_experts < 1:
            raise ValueError("n_experts must be positive")
        if self.expert_alpha <= 0 or self.gating_alpha <= 0:
            raise ValueError("expert_alpha and gating_alpha must be positive")
        if self.max_iter < 1 or self.tol <= 0:
            raise ValueError("max_iter must be positive and tol must be positive")
        X, self.x_mean_, self.x_scale_ = _standardize_fit(X)
        self.classes_, counts = np.unique(y, return_counts=True)
        if len(self.classes_) < 2 or np.any(counts < 2):
            raise ValueError("at least two samples per class are required")

        rng = np.random.default_rng(self.random_state)
        responsibilities = rng.dirichlet(np.ones(self.n_experts), size=len(X))
        gate_coefficients = np.zeros((self.n_experts, X.shape[1] + 1), dtype=float)
        gate_coefficients[:-1] = rng.normal(
            0.0, 0.05, size=(self.n_experts - 1, X.shape[1] + 1)
        )
        previous_bound = -np.inf
        self.experts_ = []
        for iteration in range(self.max_iter):
            self.experts_ = []
            for expert_index in range(self.n_experts):
                expert = LogisticRegression(
                    C=1.0 / self.expert_alpha,
                    max_iter=500,
                    solver="lbfgs",
                    random_state=None
                    if self.random_state is None
                    else int(self.random_state) + expert_index,
                )
                expert.fit(X, y, sample_weight=responsibilities[:, expert_index])
                self.experts_.append(expert)
            gate_coefficients = _fit_gate(
                X,
                responsibilities,
                self.gating_alpha,
                gate_coefficients,
                min(self.max_iter, 100),
            )
            gate_probabilities = _gate_probabilities(X, gate_coefficients)
            target_indices = np.searchsorted(self.classes_, y)
            expert_log_probabilities = np.column_stack(
                [
                    np.log(
                        np.clip(expert.predict_proba(X), np.finfo(float).tiny, 1.0)
                    )[
                        np.arange(len(X)), target_indices
                    ]
                    for expert in self.experts_
                ]
            )
            log_joint = np.log(np.clip(gate_probabilities, np.finfo(float).tiny, 1.0))
            log_joint += expert_log_probabilities
            log_evidence = logsumexp(log_joint, axis=1)
            responsibilities = np.exp(log_joint - log_evidence[:, None])
            lower_bound = float(np.mean(log_evidence))
            if abs(lower_bound - previous_bound) <= self.tol:
                break
            previous_bound = lower_bound

        self.gate_coef_ = gate_coefficients
        self.lower_bound_ = lower_bound
        self.n_iter_ = iteration + 1
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        X = _dense(X)
        X = _standardize_transform(X, self.x_mean_, self.x_scale_)
        gate_probabilities = _gate_probabilities(X, self.gate_coef_)
        expert_probabilities = np.stack(
            [expert.predict_proba(X) for expert in self.experts_], axis=0
        )
        return np.einsum("nk,knc->nc", gate_probabilities, expert_probabilities)

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class GatedLinearExpertsRegressor:
    """Mixture of ridge experts with a learned linear softmax gate."""

    def __init__(
        self,
        n_experts: int,
        expert_alpha: float,
        gating_alpha: float,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: int | None = None,
    ) -> None:
        self.n_experts = int(n_experts)
        self.expert_alpha = float(expert_alpha)
        self.gating_alpha = float(gating_alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> GatedLinearExpertsRegressor:
        X = _dense(X)
        y = np.asarray(y, dtype=float)
        if self.n_experts < 1:
            raise ValueError("n_experts must be positive")
        if self.expert_alpha <= 0 or self.gating_alpha <= 0:
            raise ValueError("expert_alpha and gating_alpha must be positive")
        if self.max_iter < 1 or self.tol <= 0:
            raise ValueError("max_iter must be positive and tol must be positive")
        X, self.x_mean_, self.x_scale_ = _standardize_fit(X)

        rng = np.random.default_rng(self.random_state)
        responsibilities = rng.dirichlet(np.ones(self.n_experts), size=len(X))
        gate_coefficients = np.zeros((self.n_experts, X.shape[1] + 1), dtype=float)
        gate_coefficients[:-1] = rng.normal(
            0.0, 0.05, size=(self.n_experts - 1, X.shape[1] + 1)
        )
        variances = np.full(self.n_experts, max(float(np.var(y)), 1e-8))
        previous_bound = -np.inf
        self.experts_ = []
        for iteration in range(self.max_iter):
            self.experts_ = []
            for expert_index in range(self.n_experts):
                expert = Ridge(alpha=self.expert_alpha)
                expert.fit(X, y, sample_weight=responsibilities[:, expert_index])
                self.experts_.append(expert)
            predictions = np.column_stack([expert.predict(X) for expert in self.experts_])
            variances = np.maximum(
                np.sum(
                    responsibilities * (y[:, None] - predictions) ** 2,
                    axis=0,
                )
                / np.maximum(np.sum(responsibilities, axis=0), 1e-12),
                1e-8,
            )
            gate_coefficients = _fit_gate(
                X,
                responsibilities,
                self.gating_alpha,
                gate_coefficients,
                min(self.max_iter, 100),
            )
            gate_probabilities = _gate_probabilities(X, gate_coefficients)
            log_joint = np.log(np.clip(gate_probabilities, np.finfo(float).tiny, 1.0))
            log_joint += -0.5 * (
                np.log(2.0 * np.pi * variances)[None, :]
                + (y[:, None] - predictions) ** 2 / variances[None, :]
            )
            log_evidence = logsumexp(log_joint, axis=1)
            responsibilities = np.exp(log_joint - log_evidence[:, None])
            lower_bound = float(np.mean(log_evidence))
            if abs(lower_bound - previous_bound) <= self.tol:
                break
            previous_bound = lower_bound

        self.gate_coef_ = gate_coefficients
        self.variances_ = variances
        self.lower_bound_ = lower_bound
        self.n_iter_ = iteration + 1
        return self

    def predict(self, X: Any) -> np.ndarray:
        X = _dense(X)
        X = _standardize_transform(X, self.x_mean_, self.x_scale_)
        gate_probabilities = _gate_probabilities(X, self.gate_coef_)
        predictions = np.column_stack([expert.predict(X) for expert in self.experts_])
        return np.sum(gate_probabilities * predictions, axis=1)
