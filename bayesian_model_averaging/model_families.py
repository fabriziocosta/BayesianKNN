"""Predictive estimators used by the model-family ensemble."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse import issparse
from scipy.special import logsumexp
from sklearn.linear_model import BayesianRidge, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


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


def make_model_estimator(
    task: str,
    model_family: str,
    neighborhood_size: int | None,
    weights: str,
    metric: str,
    covariance_structure: str | None,
) -> Any:
    """Construct one estimator for a sampled model family."""

    if model_family == "knn":
        if neighborhood_size is None:
            raise ValueError("k-NN models require a neighborhood size")
        estimator_class = KNeighborsClassifier if task == "classification" else KNeighborsRegressor
        return estimator_class(
            n_neighbors=neighborhood_size,
            weights=weights,
            metric=metric,
            n_jobs=1,
        )
    if model_family == "linear":
        if task == "classification":
            return LogisticRegression(max_iter=2000, solver="lbfgs")
        return Ridge(alpha=1.0)
    if model_family == "gaussian":
        if task == "classification":
            if covariance_structure is None:
                raise ValueError("Gaussian classifiers require a covariance structure")
            return GaussianClassifier(covariance_structure)
        return BayesianRidge()
    raise ValueError("model_family must be knn, linear, gaussian, or mixed")
