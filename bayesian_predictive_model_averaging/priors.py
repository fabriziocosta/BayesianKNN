"""Priors over ordered discrete scales."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import logsumexp


@dataclass(frozen=True)
class ParameterDraw:
    """One complete hyperparameter draw from an estimator-family prior."""

    parameters: dict[str, Any]
    log_probability: float
    metadata: dict[str, Any]

    @property
    def probability(self) -> float:
        return float(np.exp(self.log_probability))

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameters": dict(self.parameters),
            "probability": self.probability,
            "log_probability": self.log_probability,
            "metadata": dict(self.metadata),
        }


class CategoricalPrior:
    """Categorical prior over a finite collection of values."""

    def __init__(
        self,
        values: Sequence[Any],
        probabilities: Sequence[float] | None = None,
    ) -> None:
        self.values = tuple(values)
        if not self.values:
            raise ValueError("values must be non-empty")
        if probabilities is None:
            probabilities_array = np.ones(len(self.values), dtype=float)
        else:
            probabilities_array = np.asarray(probabilities, dtype=float)
            if probabilities_array.shape != (len(self.values),):
                raise ValueError("probabilities must have one value per category")
        if np.any(~np.isfinite(probabilities_array)) or np.any(probabilities_array <= 0):
            raise ValueError("probabilities must be finite and positive")
        probabilities_array /= probabilities_array.sum()
        self.probabilities = tuple(float(value) for value in probabilities_array)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"values": self.values, "probabilities": self.probabilities}

    def draw(self, rng: np.random.Generator) -> tuple[Any, float, dict[str, Any]]:
        index = int(rng.choice(len(self.values), p=self.probabilities))
        probability = float(self.probabilities[index])
        return (
            self.values[index],
            float(np.log(probability)),
            {"index": index, "probabilities": self.probabilities},
        )


class IntegerChoicePrior(CategoricalPrior):
    """Categorical prior specialized for a finite ordered integer choice set."""

    def __init__(
        self,
        values: Sequence[int],
        probabilities: Sequence[float] | None = None,
    ) -> None:
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
               for value in values):
            raise ValueError("values must contain integers")
        super().__init__(values, probabilities)


class SimplicityCategoricalPrior(CategoricalPrior):
    """Categorical prior with mass decreasing as declared complexity increases."""

    def __init__(
        self,
        values: Sequence[Any],
        complexities: Sequence[float],
        simplicity: float = 1.0,
    ) -> None:
        complexity_array = np.asarray(complexities, dtype=float)
        if len(values) != len(complexity_array):
            raise ValueError("complexities must have one value per category")
        if (
            not np.isfinite(simplicity)
            or simplicity <= 0
            or np.any(~np.isfinite(complexity_array))
            or np.any(complexity_array < 0)
        ):
            raise ValueError("simplicity and complexities must be finite and positive")
        probabilities = np.exp(-float(simplicity) * np.log1p(complexity_array))
        super().__init__(values, probabilities)
        self.complexities = tuple(float(value) for value in complexity_array)
        self.simplicity = float(simplicity)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "values": self.values,
            "complexities": self.complexities,
            "simplicity": self.simplicity,
        }


class LogUniformPrior:
    """Continuous log-uniform prior over a positive interval."""

    def __init__(self, low: float, high: float) -> None:
        if not np.isfinite(low) or not np.isfinite(high) or low <= 0 or high <= low:
            raise ValueError("low and high must be finite, positive, and low < high")
        self.low = float(low)
        self.high = float(high)

    def get_params(self, deep: bool = True) -> dict[str, float]:
        return {"low": self.low, "high": self.high}

    def draw(self, rng: np.random.Generator) -> tuple[float, float, dict[str, Any]]:
        value = float(np.exp(rng.uniform(np.log(self.low), np.log(self.high))))
        log_probability = -np.log(value) - np.log(np.log(self.high / self.low))
        return value, float(log_probability), {"low": self.low, "high": self.high}


@dataclass(frozen=True)
class ScalePriorDraw:
    """One draw from a discrete ordered-scale prior."""

    value: int | float
    index: int
    beta: float
    cutoff: float
    probability: float
    log_probability: float
    probabilities: tuple[float, ...]
    values: tuple[int | float, ...] = ()


@dataclass(frozen=True)
class GaussianCovarianceDraw:
    """One draw from the Gaussian covariance-complexity prior."""

    value: str
    probability: float
    log_probability: float
    probabilities: tuple[float, ...]


class LogisticScalePrior:
    """Monotone logistic prior for an ordered collection of integer values."""

    def __init__(self, beta_shape: float = 2.0, beta_scale: float = 1.0) -> None:
        if beta_shape <= 0:
            raise ValueError("beta_shape must be positive")
        if beta_scale <= 0:
            raise ValueError("beta_scale must be positive")
        self.beta_shape = float(beta_shape)
        self.beta_scale = float(beta_scale)

    def get_params(self, deep: bool = True) -> dict[str, float]:
        """Return sklearn-compatible constructor parameters."""

        return {"beta_shape": self.beta_shape, "beta_scale": self.beta_scale}

    def draw(self, values: Sequence[int], rng: np.random.Generator) -> ScalePriorDraw:
        """Draw one value and retain the full conditional distribution."""

        return _draw_logistic_scale(
            values,
            rng,
            beta_shape=self.beta_shape,
            beta_scale=self.beta_scale,
            integer_values=True,
        )


def _draw_logistic_scale(
    values: Sequence[int | float],
    rng: np.random.Generator,
    *,
    beta_shape: float,
    beta_scale: float,
    integer_values: bool,
) -> ScalePriorDraw:
    """Draw from the sigmoid prior over an ordered finite scale grid."""

    try:
        value_list = list(values)
    except TypeError as exc:
        raise ValueError("values must be a one-dimensional ordered sequence") from exc
    if not value_list:
        raise ValueError("values must be non-empty")
    if integer_values and any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
        for value in value_list
    ):
        raise ValueError("values must contain integers")
    if not integer_values and any(isinstance(value, (bool, np.bool_)) for value in value_list):
        raise ValueError("values must contain finite numbers")
    try:
        numeric_values = np.asarray(value_list, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("values must contain finite numbers") from exc
    if numeric_values.ndim != 1 or np.any(~np.isfinite(numeric_values)):
        raise ValueError("values must contain finite numbers")
    if any(left >= right for left, right in zip(numeric_values, numeric_values[1:])):
        raise ValueError("values must be strictly increasing")

    beta = float(rng.gamma(shape=beta_shape, scale=beta_scale))
    cutoff = float(rng.uniform(0.0, 1.0))
    n_values = len(value_list)
    positions = np.zeros(1, dtype=float) if n_values == 1 else np.linspace(0.0, 1.0, n_values)
    log_weights = -np.logaddexp(0.0, beta * (positions - cutoff))
    log_probabilities = log_weights - logsumexp(log_weights)
    probabilities = np.exp(log_probabilities)
    tiny = np.finfo(float).tiny
    probabilities = np.maximum(probabilities, tiny)
    probabilities /= probabilities.sum()

    index = int(rng.choice(n_values, p=probabilities))
    probability = float(probabilities[index])
    normalized_values = tuple(
        int(value) if integer_values else float(value) for value in numeric_values
    )
    return ScalePriorDraw(
        value=normalized_values[index],
        index=index,
        beta=beta,
        cutoff=cutoff,
        probability=probability,
        log_probability=float(np.log(probability)),
        probabilities=tuple(float(probability_) for probability_ in probabilities),
        values=normalized_values,
    )


class LogisticLogScalePrior:
    """Sigmoid-preferring prior over a finite logarithmic parameter sweep."""

    def __init__(
        self,
        low: float = 1e-2,
        high: float = 1e2,
        n_values: int = 5,
        beta_shape: float = 2.0,
        beta_scale: float = 1.0,
    ) -> None:
        if not np.isfinite(low) or not np.isfinite(high) or low <= 0 or high <= low:
            raise ValueError("low and high must be finite, positive, and low < high")
        if isinstance(n_values, (bool, np.bool_)) or not isinstance(n_values, (int, np.integer)):
            raise ValueError("n_values must be an integer")
        if n_values < 2:
            raise ValueError("n_values must be at least 2")
        if not np.isfinite(beta_shape) or beta_shape <= 0:
            raise ValueError("beta_shape must be finite and positive")
        if not np.isfinite(beta_scale) or beta_scale <= 0:
            raise ValueError("beta_scale must be finite and positive")
        self.low = float(low)
        self.high = float(high)
        self.n_values = int(n_values)
        self.beta_shape = float(beta_shape)
        self.beta_scale = float(beta_scale)

    @property
    def values(self) -> tuple[float, ...]:
        """Return the inclusive geometric sweep grid."""

        return tuple(float(value) for value in np.geomspace(self.low, self.high, self.n_values))

    def get_params(self, deep: bool = True) -> dict[str, float | int]:
        return {
            "low": self.low,
            "high": self.high,
            "n_values": self.n_values,
            "beta_shape": self.beta_shape,
            "beta_scale": self.beta_scale,
        }

    def draw(self, rng: np.random.Generator) -> ScalePriorDraw:
        """Draw one grid value using the sigmoid prior over log positions."""

        return _draw_logistic_scale(
            self.values,
            rng,
            beta_shape=self.beta_shape,
            beta_scale=self.beta_scale,
            integer_values=False,
        )


class GaussianCovariancePrior:
    """Simplicity prior over isotropic, diagonal, and full covariance."""

    structures = ("isotropic", "diagonal", "full")

    def __init__(self, simplicity: float = 1.0) -> None:
        if not np.isfinite(simplicity) or simplicity <= 0:
            raise ValueError("simplicity must be finite and positive")
        self.simplicity = float(simplicity)

    def get_params(self, deep: bool = True) -> dict[str, float]:
        return {"simplicity": self.simplicity}

    def draw(self, n_features: int, rng: np.random.Generator) -> GaussianCovarianceDraw:
        if isinstance(n_features, (bool, np.bool_)) or not isinstance(
            n_features, (int, np.integer)
        ) or n_features < 1:
            raise ValueError("n_features must be a positive integer")
        parameter_counts = np.array(
            [1, n_features, n_features * (n_features + 1) // 2], dtype=float
        )
        log_weights = -self.simplicity * np.log1p(parameter_counts)
        log_probabilities = log_weights - logsumexp(log_weights)
        probabilities = np.exp(log_probabilities)
        index = int(rng.choice(len(self.structures), p=probabilities))
        probability = float(probabilities[index])
        return GaussianCovarianceDraw(
            value=self.structures[index],
            probability=probability,
            log_probability=float(np.log(probability)),
            probabilities=tuple(float(value) for value in probabilities),
        )


def make_gaussian_covariance_prior(
    configuration: object | None,
) -> GaussianCovariancePrior:
    """Normalize a Gaussian covariance-prior object or configuration."""

    if configuration is None:
        return GaussianCovariancePrior()
    if isinstance(configuration, GaussianCovariancePrior):
        return configuration
    if isinstance(configuration, Mapping):
        values = dict(configuration)
        family = values.pop("family", values.pop("name", "simplicity"))
        if family not in {"simplicity", "gaussian_covariance"}:
            raise ValueError("only the Gaussian covariance simplicity prior is implemented")
        return GaussianCovariancePrior(**values)
    if callable(getattr(configuration, "draw", None)):
        return configuration  # type: ignore[return-value]
    raise TypeError("gaussian_covariance_prior must be a prior object or mapping")


def make_scale_prior(configuration: object | None) -> LogisticScalePrior:
    """Normalize an estimator prior object or logistic configuration mapping."""

    if configuration is None:
        return LogisticScalePrior()
    if isinstance(configuration, LogisticScalePrior):
        return configuration
    if isinstance(configuration, Mapping):
        values = dict(configuration)
        family = values.pop("family", values.pop("name", "logistic"))
        if family != "logistic":
            raise ValueError("only the logistic scale prior is implemented")
        return LogisticScalePrior(**values)
    if callable(getattr(configuration, "draw", None)):
        return configuration  # type: ignore[return-value]
    raise TypeError("scale_prior must be a prior object or a logistic configuration mapping")
