"""Bayesian Predictive Model Averaging classifier."""

from __future__ import annotations

from typing import Any

from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score

from .base import BayesianPredictiveModelAveragingBase


class BayesianPredictiveModelAveragingClassifier(
    ClassifierMixin, BayesianPredictiveModelAveragingBase
):
    """Classifier that averages sampled predictive models over CV scores."""

    _estimator_type = "classifier"

    def fit(self, X: Any, y: Any) -> BayesianPredictiveModelAveragingClassifier:
        self._fit_task(X, y, "classification")
        return self

    def predict(self, X: Any) -> Any:
        return self._predict(X)

    def predict_proba(self, X: Any) -> Any:
        return self._predict_proba(X)

    def score(self, X: Any, y: Any) -> float:
        return float(accuracy_score(y, self.predict(X)))
