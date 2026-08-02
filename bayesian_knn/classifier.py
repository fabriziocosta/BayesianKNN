"""Bayesian Monte Carlo k-NN classifier."""

from __future__ import annotations

from typing import Any

from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score

from .base import BayesianKNNBase


class BayesianKNNClassifier(ClassifierMixin, BayesianKNNBase):
    """Classifier that averages weighted k-NN models over sampled scales."""

    _estimator_type = "classifier"

    def fit(self, X: Any, y: Any) -> BayesianKNNClassifier:
        self._fit_task(X, y, "classification")
        return self

    def predict(self, X: Any) -> Any:
        return self._predict(X)

    def predict_proba(self, X: Any) -> Any:
        return self._predict_proba(X)

    def score(self, X: Any, y: Any) -> float:
        return float(accuracy_score(y, self.predict(X)))
