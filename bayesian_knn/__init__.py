"""Bayesian Monte Carlo model averaging for k-nearest neighbours."""

from .classifier import BayesianKNNClassifier
from .priors import LogisticScalePrior, ScalePriorDraw
from .regressor import BayesianKNNRegressor

__all__ = [
    "BayesianKNNClassifier",
    "BayesianKNNRegressor",
    "LogisticScalePrior",
    "ScalePriorDraw",
]

__version__ = "0.1.0"
