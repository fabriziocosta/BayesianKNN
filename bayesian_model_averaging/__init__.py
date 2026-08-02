"""Bayesian model averaging across k-NN, linear, and Gaussian predictors."""

from .classifier import BayesianModelAveragingClassifier
from .priors import (
    GaussianCovarianceDraw,
    GaussianCovariancePrior,
    LogisticScalePrior,
    ScalePriorDraw,
)
from .regressor import BayesianModelAveragingRegressor

__all__ = [
    "BayesianModelAveragingClassifier",
    "BayesianModelAveragingRegressor",
    "GaussianCovarianceDraw",
    "GaussianCovariancePrior",
    "LogisticScalePrior",
    "ScalePriorDraw",
]

__version__ = "0.1.0"
