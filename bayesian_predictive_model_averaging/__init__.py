"""BPMA with runtime estimator-family adapters."""

from .adapters import (
    DecisionTreeAdapter,
    EstimatorFamilyAdapter,
    FamilyRegistration,
    GaussianAdapter,
    GaussianMixtureAdapter,
    KNNAdapter,
    LinearAdapter,
    LinearMixtureAdapter,
    MLPAdapter,
    RecursivePartitionLinearAdapter,
    RecursivePartitionQDAAdapter,
    RecursivePartitionQuadraticAdapter,
    RecursivePartitionRBFAdapter,
    SamplingContext,
    default_family_registry,
)
from .classifier import BayesianPredictiveModelAveragingClassifier
from .priors import (
    CategoricalPrior,
    GaussianCovarianceDraw,
    GaussianCovariancePrior,
    IntegerChoicePrior,
    LogisticLogScalePrior,
    LogisticScalePrior,
    LogUniformPrior,
    ParameterDraw,
    ScalePriorDraw,
    SimplicityCategoricalPrior,
)
from .regressor import BayesianPredictiveModelAveragingRegressor

__all__ = [
    "BayesianPredictiveModelAveragingClassifier",
    "BayesianPredictiveModelAveragingRegressor",
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
    "LinearMixtureAdapter",
    "LogisticLogScalePrior",
    "LogisticScalePrior",
    "LogUniformPrior",
    "MLPAdapter",
    "RecursivePartitionLinearAdapter",
    "RecursivePartitionQuadraticAdapter",
    "RecursivePartitionQDAAdapter",
    "RecursivePartitionRBFAdapter",
    "ParameterDraw",
    "SamplingContext",
    "ScalePriorDraw",
    "SimplicityCategoricalPrior",
    "default_family_registry",
]

__version__ = "0.1.0"
