"""Internal model-draw records and public serialization."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .priors import ParameterDraw, ScalePriorDraw


def _scale_draw_dict(draw: ScalePriorDraw | None) -> dict[str, Any] | None:
    if draw is None:
        return None
    return {
        "value": draw.value,
        "index": draw.index,
        "beta": draw.beta,
        "cutoff": draw.cutoff,
        "probability": draw.probability,
        "log_probability": draw.log_probability,
        "probabilities": tuple(draw.probabilities),
    }


@dataclass
class ModelDraw:
    """One fitted model sampled from the complete prior."""

    family_name: str
    family_prior_probability: float
    parameters: dict[str, Any]
    parameter_prior: ParameterDraw
    subset_size: int = 0
    subset_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    subset_scale_draw: ScalePriorDraw | None = None
    log_prior: float = 0.0
    log_proposal: float = 0.0
    cv_log_pseudo_likelihood: float = 0.0
    estimator: Any = field(default=None, repr=False)
    log_importance_weight: float = 0.0
    posterior_weight: float = 0.0

    def __post_init__(self) -> None:
        self.log_importance_weight = self.cv_log_pseudo_likelihood

    def to_dict(self) -> dict[str, Any]:
        subset_draw = self.subset_scale_draw
        return {
            "family_name": self.family_name,
            "family_prior_probability": self.family_prior_probability,
            "parameters": dict(self.parameters),
            "parameter_prior": self.parameter_prior.to_dict(),
            "subset_size": self.subset_size,
            "subset_indices": self.subset_indices.copy(),
            "subset_scale_draw": _scale_draw_dict(subset_draw),
            "log_prior": self.log_prior,
            "log_proposal": self.log_proposal,
            "cv_log_pseudo_likelihood": self.cv_log_pseudo_likelihood,
            "log_importance_weight": self.log_importance_weight,
            "posterior_weight": self.posterior_weight,
        }


def aggregate_model_masses(draws: Sequence[ModelDraw]) -> dict[str, Any]:
    """Aggregate posterior shares by family and sampled parameter values.

    Parameter shares are conditional within each family, so the values under
    each family/parameter mapping sum to one. This keeps the diagnostic
    independent of any particular adapter's parameter names.
    """

    if not draws:
        raise ValueError("draws must contain at least one model")
    weights = np.asarray([draw.posterior_weight for draw in draws], dtype=float)
    if np.any(~np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("model posterior weights must be finite and non-negative")
    weights /= weights.sum()
    family_mass: dict[str, float] = {}
    family_draws: dict[str, list[tuple[ModelDraw, float]]] = {}
    for draw, weight in zip(draws, weights):
        weight = float(weight)
        family_mass[draw.family_name] = family_mass.get(draw.family_name, 0.0) + weight
        family_draws.setdefault(draw.family_name, []).append((draw, weight))

    parameter_mass: dict[str, dict[str, dict[Any, float]]] = {}
    for family_name, entries in family_draws.items():
        conditional_mass: dict[str, dict[Any, float]] = {}
        family_weight = family_mass[family_name]
        for draw, weight in entries:
            for parameter_name, value in draw.parameters.items():
                try:
                    hash(value)
                    key = value
                except TypeError:
                    key = repr(value)
                values = conditional_mass.setdefault(parameter_name, {})
                values[key] = values.get(key, 0.0) + weight / family_weight
        parameter_mass[family_name] = {
            parameter_name: dict(sorted(values.items(), key=lambda item: repr(item[0])))
            for parameter_name, values in sorted(conditional_mass.items())
        }

    return {
        "family": dict(sorted(family_mass.items())),
        "parameter": dict(sorted(parameter_mass.items())),
    }
