"""Priors over ordered discrete scales."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


@dataclass(frozen=True)
class ScalePriorDraw:
    """One draw from a discrete ordered-scale prior."""

    value: int
    index: int
    beta: float
    cutoff: float
    probability: float
    log_probability: float
    probabilities: tuple[float, ...]


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

        try:
            value_list = list(values)
        except TypeError as exc:
            raise ValueError("values must be a one-dimensional ordered sequence") from exc
        if not value_list:
            raise ValueError("values must be non-empty")
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
               for value in value_list):
            raise ValueError("values must contain integers")
        if any(left >= right for left, right in zip(value_list, value_list[1:])):
            raise ValueError("values must be strictly increasing")

        beta = float(rng.gamma(shape=self.beta_shape, scale=self.beta_scale))
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
        return ScalePriorDraw(
            value=int(value_list[index]),
            index=index,
            beta=beta,
            cutoff=cutoff,
            probability=probability,
            log_probability=float(np.log(probability)),
            probabilities=tuple(float(probability_) for probability_ in probabilities),
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
