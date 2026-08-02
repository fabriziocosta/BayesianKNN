"""Predictive estimators used by the model-family ensemble."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse import issparse
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture


def _dense(X: Any) -> np.ndarray:
    if issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=float)


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
