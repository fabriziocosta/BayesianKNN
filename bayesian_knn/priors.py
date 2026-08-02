"""Priors over ordered discrete scales."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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
