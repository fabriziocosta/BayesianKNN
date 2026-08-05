"""Estimator-family adapters and the built-in adapter registry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import BayesianRidge, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .model_families import (
    GatedLinearExpertsClassifier,
    GatedLinearExpertsRegressor,
    GaussianClassifier,
    GaussianMixtureClassifier,
)
from .priors import (
    CategoricalPrior,
    GaussianCovariancePrior,
    LogisticLogScalePrior,
    LogUniformPrior,
    ParameterDraw,
    ScalePriorDraw,
    SimplicityCategoricalPrior,
)


@dataclass(frozen=True)
class SamplingContext:
    """Dataset and CV facts available while an adapter samples parameters."""

    task: str
    n_features: int
    n_classes: int | None
    n_samples: int
    subset_size: int
    min_train_size: int
    classes: np.ndarray | None
    scale_prior: Any
    min_class_train_size: int | None = None
    min_class_distinct_train_size: int | None = None


@runtime_checkable
class EstimatorFamilyAdapter(Protocol):
    """Protocol implemented by one pluggable predictive model family."""

    name: str
    supported_tasks: frozenset[str]

    def sample_parameters(
        self,
        context: SamplingContext,
        rng: np.random.Generator,
    ) -> ParameterDraw:
        ...

    def build_estimator(
        self,
        task: str,
        parameters: Mapping[str, Any],
        random_state: int,
    ) -> BaseEstimator:
        ...

    def predictive_concentration(
        self,
        task: str,
        parameters: Mapping[str, Any],
    ) -> float:
        ...


@dataclass(frozen=True)
class FamilyRegistration:
    """An adapter together with its prior mass in the family mixture."""

    adapter: EstimatorFamilyAdapter
    prior_weight: float = 1.0

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Expose registration fields so sklearn can clone nested registries."""

        return {"adapter": self.adapter, "prior_weight": self.prior_weight}


def _scale_draw_metadata(draw: ScalePriorDraw) -> dict[str, Any]:
    return {
        "value": draw.value,
        "index": draw.index,
        "beta": draw.beta,
        "cutoff": draw.cutoff,
        "probability": draw.probability,
        "log_probability": draw.log_probability,
        "probabilities": tuple(draw.probabilities),
        "values": tuple(draw.values),
    }


def _categorical_metadata(
    value: Any,
    log_probability: float,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "value": value,
        "probability": float(np.exp(log_probability)),
        "log_probability": log_probability,
        **metadata,
    }


class KNNAdapter(BaseEstimator):
    """Adapter for weighted k-nearest-neighbour classifiers and regressors."""

    name = "knn"
    supported_tasks = frozenset({"classification", "regression"})

    def __init__(
        self,
        weights: str = "distance",
        metric: str = "euclidean",
        max_neighbors: int | None = None,
    ) -> None:
        self.weights = weights
        self.metric = metric
        self.max_neighbors = max_neighbors

    def sample_parameters(
        self,
        context: SamplingContext,
        rng: np.random.Generator,
    ) -> ParameterDraw:
        k_max = context.min_train_size
        if self.max_neighbors is not None:
            k_max = min(k_max, int(self.max_neighbors))
        if k_max < 1:
            raise ValueError("max_neighbors leaves no valid neighbourhood size")
        draw = context.scale_prior.draw(range(1, k_max + 1), rng)
        return ParameterDraw(
            parameters={
                "n_neighbors": draw.value,
                "weights": self.weights,
                "metric": self.metric,
            },
            log_probability=draw.log_probability,
            metadata={"neighborhood_scale_draw": _scale_draw_metadata(draw)},
        )

    def build_estimator(
        self,
        task: str,
        parameters: Mapping[str, Any],
        random_state: int,
    ) -> BaseEstimator:
        estimator_class = KNeighborsClassifier if task == "classification" else KNeighborsRegressor
        return estimator_class(
            n_neighbors=int(parameters["n_neighbors"]),
            weights=str(parameters["weights"]),
            metric=parameters["metric"],
            n_jobs=1,
        )

    def predictive_concentration(
        self,
        task: str,
        parameters: Mapping[str, Any],
    ) -> float:
        # The neighbourhood size already determines the k-NN probability
        # vector. Do not use it a second time as cross-family confidence;
        # that would make k-NN scores incomparable with other classifiers.
        return 1.0


class DecisionTreeAdapter(BaseEstimator):
    """Adapter for decision-tree classifiers and regressors."""

    name = "decision_tree"
    supported_tasks = frozenset({"classification", "regression"})

    def __init__(
        self,
        max_depth_values: Sequence[int | None] = (1, 2, 3, 4, 6, 8, None),
        min_samples_leaf_values: Sequence[int] = (1, 2, 4, 8, 16),
        min_samples_split_values: Sequence[int] = (2, 4, 8, 16, 32),
        simplicity: float = 1.0,
    ) -> None:
        self.max_depth_values = tuple(max_depth_values)
        self.min_samples_leaf_values = tuple(min_samples_leaf_values)
        self.min_samples_split_values = tuple(min_samples_split_values)
        self.simplicity = simplicity

    def sample_parameters(
        self,
        context: SamplingContext,
        rng: np.random.Generator,
    ) -> ParameterDraw:
        depth_complexities = [
            2**int(depth) if depth is not None else 2 ** (context.n_features + 3)
            for depth in self.max_depth_values
        ]
        depth_prior = SimplicityCategoricalPrior(
            self.max_depth_values,
            depth_complexities,
            simplicity=float(self.simplicity),
        )
        max_depth, depth_log_probability, depth_metadata = depth_prior.draw(rng)

        leaf_values = tuple(
            value
            for value in self.min_samples_leaf_values
            if isinstance(value, (int, np.integer))
            and not isinstance(value, (bool, np.bool_))
            and 1 <= int(value) <= context.subset_size
        )
        if not leaf_values:
            raise ValueError("no valid min_samples_leaf choices for the sampled subset")
        leaf_prior = SimplicityCategoricalPrior(
            leaf_values,
            leaf_values,
            simplicity=float(self.simplicity),
        )
        min_samples_leaf, leaf_log_probability, leaf_metadata = leaf_prior.draw(rng)

        split_values = tuple(
            value
            for value in self.min_samples_split_values
            if isinstance(value, (int, np.integer))
            and not isinstance(value, (bool, np.bool_))
            and 2 <= int(value) <= context.subset_size
        )
        if not split_values:
            raise ValueError("no valid min_samples_split choices for the sampled subset")
        split_prior = SimplicityCategoricalPrior(
            split_values,
            split_values,
            simplicity=float(self.simplicity),
        )
        min_samples_split, split_log_probability, split_metadata = split_prior.draw(rng)

        criterion_values = (
            ("gini", "entropy")
            if context.task == "classification"
            else ("squared_error",)
        )
        criterion, criterion_log_probability, criterion_metadata = CategoricalPrior(
            criterion_values
        ).draw(rng)
        splitter, splitter_log_probability, splitter_metadata = CategoricalPrior(
            ("best", "random")
        ).draw(rng)

        return ParameterDraw(
            parameters={
                "max_depth": None if max_depth is None else int(max_depth),
                "min_samples_leaf": int(min_samples_leaf),
                "min_samples_split": int(min_samples_split),
                "criterion": str(criterion),
                "splitter": str(splitter),
            },
            log_probability=float(
                depth_log_probability
                + leaf_log_probability
                + split_log_probability
                + criterion_log_probability
                + splitter_log_probability
            ),
            metadata={
                "max_depth": _categorical_metadata(
                    max_depth, depth_log_probability, depth_metadata
                ),
                "min_samples_leaf": _categorical_metadata(
                    min_samples_leaf, leaf_log_probability, leaf_metadata
                ),
                "min_samples_split": _categorical_metadata(
                    min_samples_split, split_log_probability, split_metadata
                ),
                "criterion": _categorical_metadata(
                    criterion, criterion_log_probability, criterion_metadata
                ),
                "splitter": _categorical_metadata(
                    splitter, splitter_log_probability, splitter_metadata
                ),
            },
        )

    def build_estimator(
        self,
        task: str,
        parameters: Mapping[str, Any],
        random_state: int,
    ) -> BaseEstimator:
        estimator_class = (
            DecisionTreeClassifier if task == "classification" else DecisionTreeRegressor
        )
        return estimator_class(
            max_depth=parameters["max_depth"],
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            min_samples_split=int(parameters["min_samples_split"]),
            criterion=str(parameters["criterion"]),
            splitter=str(parameters["splitter"]),
            random_state=random_state,
        )

    def predictive_concentration(
        self,
        task: str,
        parameters: Mapping[str, Any],
    ) -> float:
        return 1.0


class RandomForestAdapter(BaseEstimator):
    """Adapter for random-forest classifiers and regressors."""

    name = "random_forest"
    supported_tasks = frozenset({"classification", "regression"})

    def __init__(
        self,
        n_estimators_values: Sequence[int] = (100, 200, 400),
        max_depth_values: Sequence[int | None] = (4, 6, 8, None),
        min_samples_leaf_values: Sequence[int] = (1, 2, 4),
        min_samples_split_values: Sequence[int] = (2, 4, 8, 16, 32),
        max_features_values: Sequence[str | float] = (1.0,),
        simplicity: float = 1.0,
    ) -> None:
        self.n_estimators_values = tuple(n_estimators_values)
        self.max_depth_values = tuple(max_depth_values)
        self.min_samples_leaf_values = tuple(min_samples_leaf_values)
        self.min_samples_split_values = tuple(min_samples_split_values)
        self.max_features_values = tuple(max_features_values)
        self.simplicity = simplicity

    def sample_parameters(
        self,
        context: SamplingContext,
        rng: np.random.Generator,
    ) -> ParameterDraw:
        estimator_values = tuple(
            value
            for value in self.n_estimators_values
            if isinstance(value, (int, np.integer))
            and not isinstance(value, (bool, np.bool_))
            and int(value) >= 1
        )
        if not estimator_values:
            raise ValueError("no valid n_estimators choices")
        n_estimators, n_estimators_log_probability, n_estimators_metadata = (
            SimplicityCategoricalPrior(
                estimator_values,
                estimator_values,
                simplicity=float(self.simplicity),
            ).draw(rng)
        )

        depth_complexities = [
            2**int(depth) if depth is not None else 2 ** (context.n_features + 3)
            for depth in self.max_depth_values
        ]
        max_depth, depth_log_probability, depth_metadata = SimplicityCategoricalPrior(
            self.max_depth_values,
            depth_complexities,
            simplicity=float(self.simplicity),
        ).draw(rng)

        leaf_values = tuple(
            value
            for value in self.min_samples_leaf_values
            if isinstance(value, (int, np.integer))
            and not isinstance(value, (bool, np.bool_))
            and 1 <= int(value) <= context.subset_size
        )
        if not leaf_values:
            raise ValueError("no valid min_samples_leaf choices for the sampled subset")
        min_samples_leaf, leaf_log_probability, leaf_metadata = SimplicityCategoricalPrior(
            leaf_values,
            leaf_values,
            simplicity=float(self.simplicity),
        ).draw(rng)

        split_values = tuple(
            value
            for value in self.min_samples_split_values
            if isinstance(value, (int, np.integer))
            and not isinstance(value, (bool, np.bool_))
            and 2 <= int(value) <= context.subset_size
        )
        if not split_values:
            raise ValueError("no valid min_samples_split choices for the sampled subset")
        min_samples_split, split_log_probability, split_metadata = SimplicityCategoricalPrior(
            split_values,
            split_values,
            simplicity=float(self.simplicity),
        ).draw(rng)

        criterion_values = (
            ("gini", "entropy")
            if context.task == "classification"
            else ("squared_error",)
        )
        criterion, criterion_log_probability, criterion_metadata = CategoricalPrior(
            criterion_values
        ).draw(rng)
        max_features, max_features_log_probability, max_features_metadata = CategoricalPrior(
            self.max_features_values
        ).draw(rng)

        return ParameterDraw(
            parameters={
                "n_estimators": int(n_estimators),
                "max_depth": None if max_depth is None else int(max_depth),
                "min_samples_leaf": int(min_samples_leaf),
                "min_samples_split": int(min_samples_split),
                "criterion": str(criterion),
                "max_features": max_features,
            },
            log_probability=float(
                n_estimators_log_probability
                + depth_log_probability
                + leaf_log_probability
                + split_log_probability
                + criterion_log_probability
                + max_features_log_probability
            ),
            metadata={
                "n_estimators": _categorical_metadata(
                    n_estimators, n_estimators_log_probability, n_estimators_metadata
                ),
                "max_depth": _categorical_metadata(
                    max_depth, depth_log_probability, depth_metadata
                ),
                "min_samples_leaf": _categorical_metadata(
                    min_samples_leaf, leaf_log_probability, leaf_metadata
                ),
                "min_samples_split": _categorical_metadata(
                    min_samples_split, split_log_probability, split_metadata
                ),
                "criterion": _categorical_metadata(
                    criterion, criterion_log_probability, criterion_metadata
                ),
                "max_features": _categorical_metadata(
                    max_features, max_features_log_probability, max_features_metadata
                ),
            },
        )

    def build_estimator(
        self,
        task: str,
        parameters: Mapping[str, Any],
        random_state: int,
    ) -> BaseEstimator:
        estimator_class = (
            RandomForestClassifier if task == "classification" else RandomForestRegressor
        )
        return estimator_class(
            n_estimators=int(parameters["n_estimators"]),
            max_depth=parameters["max_depth"],
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            min_samples_split=int(parameters["min_samples_split"]),
            criterion=str(parameters["criterion"]),
            max_features=parameters["max_features"],
            n_jobs=1,
            random_state=random_state,
        )

    def predictive_concentration(
        self,
        task: str,
        parameters: Mapping[str, Any],
    ) -> float:
        return 1.0


class LinearAdapter(BaseEstimator):
    """Adapter for logistic classification and ridge regression."""

    name = "linear"
    supported_tasks = frozenset({"classification", "regression"})

    def sample_parameters(
        self,
        context: SamplingContext,
        rng: np.random.Generator,
    ) -> ParameterDraw:
        if context.task == "classification":
            parameters = {"solver": "lbfgs", "max_iter": 2000}
        else:
            parameters = {"alpha": 1.0}
        return ParameterDraw(parameters=parameters, log_probability=0.0, metadata={})

    def build_estimator(
        self,
        task: str,
        parameters: Mapping[str, Any],
        random_state: int,
    ) -> BaseEstimator:
        if task == "classification":
            return LogisticRegression(
                solver=str(parameters["solver"]),
                max_iter=int(parameters["max_iter"]),
                random_state=random_state,
            )
        return Ridge(alpha=float(parameters["alpha"]))

    def predictive_concentration(
        self,
        task: str,
        parameters: Mapping[str, Any],
    ) -> float:
        return 1.0


def _recursive_partition_components() -> tuple[Any, Any, Any]:
    """Load the optional recursive-partition classifier implementation."""

    try:
        from recursive_partition import EqualPriorQDA, RecursivePartitionClassifier
        from sklearn.svm import SVC
    except ImportError as error:
        raise ImportError(
            "RecursivePartition*Adapter requires the optional "
            "'recursive-partition-classifier' package. Install the sister "
            "repository or add it to PYTHONPATH."
        ) from error
    return RecursivePartitionClassifier, EqualPriorQDA, SVC


class _RecursivePartitionSVCAdapter(BaseEstimator):
    """Shared simplicity-prior implementation for recursive SVM adapters."""

    supported_tasks = frozenset({"classification"})

    def __init__(
        self,
        kernel: str,
        *,
        degree: int = 3,
        coef0: float = 0.0,
        gamma: str | float = "scale",
        c_values: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
        c_prior: LogisticLogScalePrior | None = None,
        simplicity: float = 1.0,
        class_weight: str | dict[Any, float] | None = "balanced",
        probability_mode: str = "leaf_frequency",
        probability_smoothing: float = 1.0,
        on_fit_failure: str = "leaf",
        max_depth: int | None = None,
        max_nodes: int | None = None,
    ) -> None:
        self.kernel = kernel
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma
        self.c_values = tuple(c_values)
        self.c_prior = c_prior
        self.simplicity = simplicity
        self.class_weight = class_weight
        self.probability_mode = probability_mode
        self.probability_smoothing = probability_smoothing
        self.on_fit_failure = on_fit_failure
        self.max_depth = max_depth
        self.max_nodes = max_nodes

    def sample_parameters(
        self,
        context: SamplingContext,
        rng: np.random.Generator,
    ) -> ParameterDraw:
        if context.task != "classification":
            raise ValueError(f"{self.name} supports classification only")
        c_values = tuple(float(value) for value in self.c_values)
        if self.c_prior is None and (
            not c_values
            or any(not np.isfinite(value) or value <= 0 for value in c_values)
            or len(set(c_values)) != len(c_values)
        ):
            raise ValueError("c_values must contain distinct, finite, positive values")
        if not np.isfinite(self.simplicity) or self.simplicity <= 0:
            raise ValueError("simplicity must be finite and positive")

        # Lower C means stronger margin regularization and a simpler SVM. A
        # LogisticLogScalePrior sweeps a geometric grid while retaining the
        # same sigmoid preference for lower values.
        if self.c_prior is None:
            c_prior = SimplicityCategoricalPrior(
                c_values,
                complexities=c_values,
                simplicity=float(self.simplicity),
            )
            c_value, c_log_probability, c_metadata = c_prior.draw(rng)
        else:
            c_draw = self.c_prior.draw(rng)
            c_value = c_draw.value
            c_log_probability = c_draw.log_probability
            c_metadata = _scale_draw_metadata(c_draw)
        return ParameterDraw(
            parameters={
                "C": float(c_value),
                "kernel": self.kernel,
                "degree": int(self.degree),
                "coef0": float(self.coef0),
                "gamma": self.gamma,
                "class_weight": self.class_weight,
            },
            log_probability=float(c_log_probability),
            metadata={
                "C": _categorical_metadata(c_value, c_log_probability, c_metadata),
            },
        )

    def build_estimator(
        self,
        task: str,
        parameters: Mapping[str, Any],
        random_state: int,
    ) -> BaseEstimator:
        if task != "classification":
            raise ValueError(f"{self.name} supports classification only")
        RecursivePartitionClassifier, _, SVC = _recursive_partition_components()
        base_estimator = SVC(
            C=float(parameters["C"]),
            kernel=str(parameters.get("kernel", self.kernel)),
            degree=int(parameters.get("degree", self.degree)),
            coef0=float(parameters.get("coef0", self.coef0)),
            gamma=parameters.get("gamma", self.gamma),
            class_weight=parameters.get("class_weight", self.class_weight),
            random_state=random_state,
        )
        return RecursivePartitionClassifier(
            base_estimator=base_estimator,
            probability_mode=self.probability_mode,
            probability_smoothing=float(self.probability_smoothing),
            on_fit_failure=self.on_fit_failure,
            max_depth=self.max_depth,
            max_nodes=self.max_nodes,
        )

    def predictive_concentration(
        self,
        task: str,
        parameters: Mapping[str, Any],
    ) -> float:
        return 1.0


class RecursivePartitionLinearAdapter(_RecursivePartitionSVCAdapter):
    """Adapter for a non-bagged recursive partitioner driven by a linear SVM."""

    name = "recursive_linear"

    def __init__(
        self,
        c_values: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
        c_prior: LogisticLogScalePrior | None = None,
        simplicity: float = 1.0,
        class_weight: str | dict[Any, float] | None = "balanced",
        probability_mode: str = "leaf_frequency",
        probability_smoothing: float = 1.0,
        on_fit_failure: str = "leaf",
        max_depth: int | None = None,
        max_nodes: int | None = None,
    ) -> None:
        super().__init__(
            "linear",
            c_values=c_values,
            c_prior=c_prior,
            simplicity=simplicity,
            class_weight=class_weight,
            probability_mode=probability_mode,
            probability_smoothing=probability_smoothing,
            on_fit_failure=on_fit_failure,
            max_depth=max_depth,
            max_nodes=max_nodes,
        )


class RecursivePartitionQuadraticAdapter(_RecursivePartitionSVCAdapter):
    """Adapter for a non-bagged recursive partitioner driven by a degree-two SVM."""

    name = "recursive_quadratic"

    def __init__(
        self,
        c_values: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
        c_prior: LogisticLogScalePrior | None = None,
        simplicity: float = 1.0,
        gamma: str | float = "scale",
        class_weight: str | dict[Any, float] | None = "balanced",
        probability_mode: str = "leaf_frequency",
        probability_smoothing: float = 1.0,
        on_fit_failure: str = "leaf",
        max_depth: int | None = None,
        max_nodes: int | None = None,
    ) -> None:
        super().__init__(
            "poly",
            degree=2,
            coef0=1.0,
            gamma=gamma,
            c_values=c_values,
            c_prior=c_prior,
            simplicity=simplicity,
            class_weight=class_weight,
            probability_mode=probability_mode,
            probability_smoothing=probability_smoothing,
            on_fit_failure=on_fit_failure,
            max_depth=max_depth,
            max_nodes=max_nodes,
        )


class RecursivePartitionRBFAdapter(_RecursivePartitionSVCAdapter):
    """Adapter for a non-bagged recursive partitioner driven by an RBF SVM."""

    name = "recursive_rbf"

    def __init__(
        self,
        c_values: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
        c_prior: LogisticLogScalePrior | None = None,
        simplicity: float = 1.0,
        gamma: str | float = "scale",
        class_weight: str | dict[Any, float] | None = "balanced",
        probability_mode: str = "leaf_frequency",
        probability_smoothing: float = 1.0,
        on_fit_failure: str = "leaf",
        max_depth: int | None = None,
        max_nodes: int | None = None,
    ) -> None:
        super().__init__(
            "rbf",
            gamma=gamma,
            c_values=c_values,
            c_prior=c_prior,
            simplicity=simplicity,
            class_weight=class_weight,
            probability_mode=probability_mode,
            probability_smoothing=probability_smoothing,
            on_fit_failure=on_fit_failure,
            max_depth=max_depth,
            max_nodes=max_nodes,
        )


class RecursivePartitionQDAAdapter(BaseEstimator):
    """Adapter for a non-bagged recursive partitioner driven by equal-prior QDA."""

    name = "recursive_qda"
    supported_tasks = frozenset({"classification"})

    def __init__(
        self,
        reg_param: float = 0.05,
        probability_mode: str = "leaf_frequency",
        probability_smoothing: float = 1.0,
        on_fit_failure: str = "leaf",
        max_depth: int | None = None,
        max_nodes: int | None = None,
    ) -> None:
        self.reg_param = reg_param
        self.probability_mode = probability_mode
        self.probability_smoothing = probability_smoothing
        self.on_fit_failure = on_fit_failure
        self.max_depth = max_depth
        self.max_nodes = max_nodes

    def sample_parameters(
        self,
        context: SamplingContext,
        rng: np.random.Generator,
    ) -> ParameterDraw:
        if context.task != "classification":
            raise ValueError(f"{self.name} supports classification only")
        return ParameterDraw(
            parameters={"reg_param": float(self.reg_param)},
            log_probability=0.0,
            metadata={},
        )

    def build_estimator(
        self,
        task: str,
        parameters: Mapping[str, Any],
        random_state: int,
    ) -> BaseEstimator:
        if task != "classification":
            raise ValueError(f"{self.name} supports classification only")
        RecursivePartitionClassifier, EqualPriorQDA, _ = _recursive_partition_components()
        return RecursivePartitionClassifier(
            base_estimator=EqualPriorQDA(
                reg_param=float(parameters.get("reg_param", self.reg_param))
            ),
            probability_mode=self.probability_mode,
            probability_smoothing=float(self.probability_smoothing),
            on_fit_failure=self.on_fit_failure,
            max_depth=self.max_depth,
            max_nodes=self.max_nodes,
        )

    def predictive_concentration(
        self,
        task: str,
        parameters: Mapping[str, Any],
    ) -> float:
        return 1.0


class LinearMixtureAdapter(BaseEstimator):
    """Adapter for gated mixtures of logistic or ridge linear experts."""

    name = "linear_mixture"
    supported_tasks = frozenset({"classification", "regression"})

    def __init__(
        self,
        max_experts: int = 5,
        expert_simplicity: float = 1.0,
        expert_alpha_prior: LogUniformPrior | None = None,
        gating_alpha_prior: LogUniformPrior | None = None,
        max_iter: int = 100,
        tol: float = 1e-4,
    ) -> None:
        self.max_experts = max_experts
        self.expert_simplicity = expert_simplicity
        self.expert_alpha_prior = expert_alpha_prior
        self.gating_alpha_prior = gating_alpha_prior
        self.max_iter = max_iter
        self.tol = tol

    def sample_parameters(
        self,
        context: SamplingContext,
        rng: np.random.Generator,
    ) -> ParameterDraw:
        max_experts = int(self.max_experts)
        if max_experts < 1:
            raise ValueError("max_experts must be positive")
        if not np.isfinite(self.expert_simplicity) or self.expert_simplicity <= 0:
            raise ValueError("expert_simplicity must be finite and positive")
        valid_max_experts = min(max_experts, context.min_train_size)
        if valid_max_experts < 1:
            raise ValueError("no valid linear-expert count")
        expert_count_prior = SimplicityCategoricalPrior(
            range(1, valid_max_experts + 1),
            range(1, valid_max_experts + 1),
            simplicity=float(self.expert_simplicity),
        )
        n_experts, count_log_probability, count_metadata = expert_count_prior.draw(rng)

        expert_prior = self.expert_alpha_prior or LogUniformPrior(1e-3, 1e1)
        gating_prior = self.gating_alpha_prior or LogUniformPrior(1e-3, 1e1)
        expert_alpha, expert_log_probability, expert_metadata = expert_prior.draw(rng)
        gating_alpha, gating_log_probability, gating_metadata = gating_prior.draw(rng)
        return ParameterDraw(
            parameters={
                "n_experts": int(n_experts),
                "expert_alpha": expert_alpha,
                "gating_alpha": gating_alpha,
                "max_iter": int(self.max_iter),
                "tol": float(self.tol),
            },
            log_probability=float(
                count_log_probability + expert_log_probability + gating_log_probability
            ),
            metadata={
                "expert_count_draw": _categorical_metadata(
                    n_experts, count_log_probability, count_metadata
                ),
                "expert_alpha": {
                    "value": expert_alpha,
                    "log_probability": expert_log_probability,
                    **expert_metadata,
                },
                "gating_alpha": {
                    "value": gating_alpha,
                    "log_probability": gating_log_probability,
                    **gating_metadata,
                },
                "valid_max_experts": valid_max_experts,
            },
        )

    def build_estimator(
        self,
        task: str,
        parameters: Mapping[str, Any],
        random_state: int,
    ) -> BaseEstimator:
        estimator_class = (
            GatedLinearExpertsClassifier
            if task == "classification"
            else GatedLinearExpertsRegressor
        )
        return estimator_class(
            n_experts=int(parameters["n_experts"]),
            expert_alpha=float(parameters["expert_alpha"]),
            gating_alpha=float(parameters["gating_alpha"]),
            max_iter=int(parameters["max_iter"]),
            tol=float(parameters["tol"]),
            random_state=random_state,
        )

    def predictive_concentration(
        self,
        task: str,
        parameters: Mapping[str, Any],
    ) -> float:
        return 1.0


class GaussianAdapter(BaseEstimator):
    """Adapter for Gaussian classification and Bayesian ridge regression."""

    name = "gaussian"
    supported_tasks = frozenset({"classification", "regression"})

    def __init__(self, covariance_prior: GaussianCovariancePrior | None = None) -> None:
        self.covariance_prior = covariance_prior

    def sample_parameters(
        self,
        context: SamplingContext,
        rng: np.random.Generator,
    ) -> ParameterDraw:
        if context.task != "classification":
            return ParameterDraw(parameters={}, log_probability=0.0, metadata={})
        prior = self.covariance_prior or GaussianCovariancePrior()
        draw = prior.draw(context.n_features, rng)
        return ParameterDraw(
            parameters={"covariance_structure": draw.value},
            log_probability=draw.log_probability,
            metadata={
                "covariance_draw": {
                    "value": draw.value,
                    "probability": draw.probability,
                    "log_probability": draw.log_probability,
                    "probabilities": tuple(draw.probabilities),
                }
            },
        )

    def build_estimator(
        self,
        task: str,
        parameters: Mapping[str, Any],
        random_state: int,
    ) -> BaseEstimator:
        if task == "classification":
            return GaussianClassifier(str(parameters["covariance_structure"]))
        return BayesianRidge()

    def predictive_concentration(
        self,
        task: str,
        parameters: Mapping[str, Any],
    ) -> float:
        return 1.0


class GaussianMixtureAdapter(BaseEstimator):
    """Adapter for class-conditional Gaussian-mixture classifiers."""

    name = "gaussian_mixture"
    supported_tasks = frozenset({"classification"})

    def __init__(
        self,
        max_components: int = 30,
        component_simplicity: float = 1.0,
        covariance_prior: GaussianCovariancePrior | None = None,
        reg_covar: float = 1e-6,
        max_iter: int = 100,
    ) -> None:
        self.max_components = max_components
        self.component_simplicity = component_simplicity
        self.covariance_prior = covariance_prior
        self.reg_covar = reg_covar
        self.max_iter = max_iter

    def sample_parameters(
        self,
        context: SamplingContext,
        rng: np.random.Generator,
    ) -> ParameterDraw:
        if context.task != "classification":
            raise ValueError("GaussianMixtureAdapter supports classification only")
        max_components = int(self.max_components)
        if max_components < 1:
            raise ValueError("max_components must be positive")
        if not np.isfinite(self.component_simplicity) or self.component_simplicity <= 0:
            raise ValueError("component_simplicity must be finite and positive")
        if context.min_class_train_size is None:
            if context.n_classes is None or context.n_classes < 1:
                raise ValueError("Gaussian mixture sampling requires class counts")
            min_class_train_size = max(1, context.min_train_size // context.n_classes)
        else:
            min_class_train_size = int(context.min_class_train_size)
        if context.min_class_distinct_train_size is None:
            min_class_distinct_train_size = min_class_train_size
        else:
            min_class_distinct_train_size = int(context.min_class_distinct_train_size)
        valid_max_components = min(
            max_components,
            min_class_train_size,
            min_class_distinct_train_size,
        )
        if valid_max_components < 1:
            raise ValueError("no valid Gaussian-mixture component count")

        component_values = tuple(range(1, valid_max_components + 1))
        component_prior = SimplicityCategoricalPrior(
            component_values,
            component_values,
            simplicity=float(self.component_simplicity),
        )
        n_components, component_log_probability, component_metadata = component_prior.draw(rng)

        covariance_prior = self.covariance_prior or GaussianCovariancePrior()
        covariance_draw = covariance_prior.draw(context.n_features, rng)
        return ParameterDraw(
            parameters={
                "n_components": int(n_components),
                "covariance_structure": covariance_draw.value,
                "reg_covar": float(self.reg_covar),
                "max_iter": int(self.max_iter),
            },
            log_probability=float(component_log_probability + covariance_draw.log_probability),
            metadata={
                "component_draw": _categorical_metadata(
                    n_components, component_log_probability, component_metadata
                ),
                "covariance_draw": {
                    "value": covariance_draw.value,
                    "probability": covariance_draw.probability,
                    "log_probability": covariance_draw.log_probability,
                    "probabilities": tuple(covariance_draw.probabilities),
                },
                "valid_max_components": valid_max_components,
            },
        )

    def build_estimator(
        self,
        task: str,
        parameters: Mapping[str, Any],
        random_state: int,
    ) -> BaseEstimator:
        if task != "classification":
            raise ValueError("GaussianMixtureAdapter supports classification only")
        return GaussianMixtureClassifier(
            n_components=int(parameters["n_components"]),
            covariance_structure=str(parameters["covariance_structure"]),
            reg_covar=float(parameters.get("reg_covar", self.reg_covar)),
            max_iter=int(parameters.get("max_iter", self.max_iter)),
            random_state=random_state,
        )

    def predictive_concentration(
        self,
        task: str,
        parameters: Mapping[str, Any],
    ) -> float:
        return 1.0


class MLPAdapter(BaseEstimator):
    """Adapter with simplicity priors over small neural-network architectures."""

    name = "mlp"
    supported_tasks = frozenset({"classification", "regression"})

    def __init__(
        self,
        hidden_layer_sizes: Sequence[tuple[int, ...]] | None = None,
        activations: Sequence[str] = ("relu", "tanh", "logistic"),
        alpha_prior: LogUniformPrior | None = None,
        learning_rate_prior: LogUniformPrior | None = None,
        max_iter: int = 500,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activations = tuple(activations)
        self.alpha_prior = alpha_prior
        self.learning_rate_prior = learning_rate_prior
        self.max_iter = max_iter

    def _architecture_prior(self) -> CategoricalPrior:
        architectures = self.hidden_layer_sizes or ((16,), (32,), (64,), (32, 16))
        complexities = [sum(architecture) for architecture in architectures]
        return SimplicityCategoricalPrior(architectures, complexities)

    def sample_parameters(
        self,
        context: SamplingContext,
        rng: np.random.Generator,
    ) -> ParameterDraw:
        architecture_prior = self._architecture_prior()
        architecture, architecture_log_probability, architecture_metadata = architecture_prior.draw(
            rng
        )
        activation_prior = CategoricalPrior(self.activations)
        activation, activation_log_probability, activation_metadata = activation_prior.draw(rng)
        alpha_prior = self.alpha_prior or LogUniformPrior(1e-6, 1e-1)
        learning_rate_prior = self.learning_rate_prior or LogUniformPrior(1e-4, 1e-1)
        alpha, alpha_log_probability, alpha_metadata = alpha_prior.draw(rng)
        learning_rate, learning_rate_log_probability, learning_rate_metadata = (
            learning_rate_prior.draw(rng)
        )
        return ParameterDraw(
            parameters={
                "hidden_layer_sizes": tuple(architecture),
                "activation": str(activation),
                "alpha": alpha,
                "learning_rate_init": learning_rate,
                "max_iter": int(self.max_iter),
                "solver": "adam",
            },
            log_probability=float(
                architecture_log_probability
                + activation_log_probability
                + alpha_log_probability
                + learning_rate_log_probability
            ),
            metadata={
                "hidden_layer_sizes": _categorical_metadata(
                    architecture, architecture_log_probability, architecture_metadata
                ),
                "activation": _categorical_metadata(
                    activation, activation_log_probability, activation_metadata
                ),
                "alpha": {
                    "value": alpha,
                    "log_probability": alpha_log_probability,
                    **alpha_metadata,
                },
                "learning_rate_init": {
                    "value": learning_rate,
                    "log_probability": learning_rate_log_probability,
                    **learning_rate_metadata,
                },
            },
        )

    def build_estimator(
        self,
        task: str,
        parameters: Mapping[str, Any],
        random_state: int,
    ) -> BaseEstimator:
        estimator_class = MLPClassifier if task == "classification" else MLPRegressor
        return estimator_class(
            hidden_layer_sizes=tuple(parameters["hidden_layer_sizes"]),
            activation=str(parameters["activation"]),
            alpha=float(parameters["alpha"]),
            learning_rate_init=float(parameters["learning_rate_init"]),
            max_iter=int(parameters["max_iter"]),
            solver=str(parameters["solver"]),
            random_state=random_state,
        )

    def predictive_concentration(
        self,
        task: str,
        parameters: Mapping[str, Any],
    ) -> float:
        return 1.0


def default_family_registry() -> tuple[FamilyRegistration, ...]:
    """Return the default built-in family mixture.

    The entries use equal relative weights. The registry normalizer converts
    those weights to a uniform prior over the five default families.
    """

    return tuple(
        FamilyRegistration(adapter, prior_weight=1.0)
        for adapter in (
            KNNAdapter(),
            LinearMixtureAdapter(),
            GaussianMixtureAdapter(),
            MLPAdapter(),
            RandomForestAdapter(),
        )
    )


def normalize_family_registry(
    registry: Sequence[FamilyRegistration | EstimatorFamilyAdapter] | None,
) -> tuple[FamilyRegistration, ...]:
    """Validate and normalize a user-supplied runtime family registry."""

    registrations = default_family_registry() if registry is None else tuple(registry)
    if not registrations:
        raise ValueError("family_registry must contain at least one adapter")
    normalized: list[FamilyRegistration] = []
    names: set[str] = set()
    for item in registrations:
        registration = item if isinstance(item, FamilyRegistration) else FamilyRegistration(item)
        adapter = registration.adapter
        name = getattr(adapter, "name", None)
        if not isinstance(name, str) or not name:
            raise ValueError("each adapter must define a non-empty string name")
        if name in names:
            raise ValueError(f"duplicate estimator-family adapter name: {name}")
        if not getattr(adapter, "supported_tasks", None):
            raise ValueError(f"adapter {name!r} must declare supported_tasks")
        weight = float(registration.prior_weight)
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("family prior weights must be finite and positive")
        names.add(name)
        normalized.append(FamilyRegistration(adapter, weight))
    total = sum(registration.prior_weight for registration in normalized)
    return tuple(
        FamilyRegistration(registration.adapter, registration.prior_weight / total)
        for registration in normalized
    )
