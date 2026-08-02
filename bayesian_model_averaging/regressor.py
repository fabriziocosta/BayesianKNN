"""Bayesian model-averaging regressor."""

from __future__ import annotations

from typing import Any

from sklearn.base import RegressorMixin
from sklearn.metrics import r2_score

from .base import BayesianModelAveragingBase


class BayesianModelAveragingRegressor(RegressorMixin, BayesianModelAveragingBase):
    """Regressor that averages sampled predictive models over CV scores."""

    _estimator_type = "regressor"

    def fit(self, X: Any, y: Any) -> BayesianModelAveragingRegressor:
        self._fit_task(X, y, "regression")
        return self

    def predict(self, X: Any) -> Any:
        return self._predict(X)

    def score(self, X: Any, y: Any) -> float:
        return float(r2_score(y, self.predict(X)))
