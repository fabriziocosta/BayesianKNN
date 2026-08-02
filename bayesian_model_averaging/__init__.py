"""Bayesian model averaging with runtime estimator-family adapters."""

from .adapters import (
    DecisionTreeAdapter,
    EstimatorFamilyAdapter,
    FamilyRegistration,
    GaussianAdapter,
    GaussianMixtureAdapter,
    KNNAdapter,
    LinearAdapter,
    MLPAdapter,
    SamplingContext,
    default_family_registry,
)
from .classifier import BayesianModelAveragingClassifier
from .priors import (
    CategoricalPrior,
    GaussianCovarianceDraw,
    GaussianCovariancePrior,
    IntegerChoicePrior,
    LogisticScalePrior,
    LogUniformPrior,
    ParameterDraw,
    ScalePriorDraw,
    SimplicityCategoricalPrior,
)
from .regressor import BayesianModelAveragingRegressor

__all__ = [
    "BayesianModelAveragingClassifier",
    "BayesianModelAveragingRegressor",
    "CategoricalPrior",
    "DecisionTreeAdapter",
    "EstimatorFamilyAdapter",
    "FamilyRegistration",
    "GaussianAdapter",
    "GaussianCovarianceDraw",
    "GaussianCovariancePrior",
    "GaussianMixtureAdapter",
    "IntegerChoicePrior",
    "KNNAdapter",
    "LinearAdapter",
    "LogUniformPrior",
    "LogisticScalePrior",
    "MLPAdapter",
    "ParameterDraw",
    "SamplingContext",
    "ScalePriorDraw",
    "SimplicityCategoricalPrior",
    "default_family_registry",
]

__version__ = "0.1.0"
