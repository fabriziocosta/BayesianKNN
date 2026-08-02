"""Internal model-draw records and public serialization."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .priors import GaussianCovarianceDraw, ScalePriorDraw


def _draw_dict(draw: ScalePriorDraw) -> dict[str, Any]:
    return {
        "value": draw.value,
        "index": draw.index,
        "beta": draw.beta,
        "cutoff": draw.cutoff,
        "probability": draw.probability,
        "log_probability": draw.log_probability,
        "probabilities": tuple(draw.probabilities),
    }


def _covariance_draw_dict(draw: GaussianCovarianceDraw | None) -> dict[str, Any] | None:
    if draw is None:
        return None
    return {
        "value": draw.value,
        "probability": draw.probability,
        "log_probability": draw.log_probability,
        "probabilities": tuple(draw.probabilities),
    }


@dataclass
class ModelDraw:
    model_family: str
    representation_family: str
    projection_dimension: int
    projection_parameters: dict[str, Any]
    representation_object: Any = field(repr=False)
    model_family_probability: float = 1.0
    representation_family_probability: float = 1.0
    gaussian_covariance_structure: str | None = None
    gaussian_covariance_probability: float = 1.0
    gaussian_covariance_draw: GaussianCovarianceDraw | None = field(default=None, repr=False)
    subset_size: int = 0
    subset_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    neighborhood_size: int | None = None
    projection_scale_draw: ScalePriorDraw | None = None
    subset_scale_draw: ScalePriorDraw | None = None
    neighbor_scale_draw: ScalePriorDraw | None = None
    log_prior: float = 0.0
    log_proposal: float = 0.0
    cv_log_pseudo_likelihood: float = 0.0
    estimator: Any = field(default=None, repr=False)
    log_importance_weight: float = 0.0
    posterior_weight: float = 0.0

    def __post_init__(self) -> None:
        self.log_importance_weight = self.cv_log_pseudo_likelihood

    def to_dict(self) -> dict[str, Any]:
        projection_draw = self.projection_scale_draw
        subset_draw = self.subset_scale_draw
        neighbor_draw = self.neighbor_scale_draw
        return {
            "model_family": self.model_family,
            "model_family_probability": self.model_family_probability,
            "representation_family": self.representation_family,
            "representation_family_probability": self.representation_family_probability,
            "gaussian_covariance_structure": self.gaussian_covariance_structure,
            "gaussian_covariance_probability": self.gaussian_covariance_probability,
            "gaussian_covariance_draw": _covariance_draw_dict(
                self.gaussian_covariance_draw
            ),
            "projection_dimension": self.projection_dimension,
            "projection_parameters": dict(self.projection_parameters),
            "subset_size": self.subset_size,
            "subset_indices": self.subset_indices.copy(),
            "neighborhood_size": self.neighborhood_size,
            "projection_scale_draw": (
                None if projection_draw is None else _draw_dict(projection_draw)
            ),
            "subset_scale_draw": None if subset_draw is None else _draw_dict(subset_draw),
            "neighbor_scale_draw": None if neighbor_draw is None else _draw_dict(neighbor_draw),
            "projection_beta": None if projection_draw is None else projection_draw.beta,
            "projection_cutoff": None if projection_draw is None else projection_draw.cutoff,
            "subset_beta": None if subset_draw is None else subset_draw.beta,
            "subset_cutoff": None if subset_draw is None else subset_draw.cutoff,
            "neighbor_beta": None if neighbor_draw is None else neighbor_draw.beta,
            "neighbor_cutoff": None if neighbor_draw is None else neighbor_draw.cutoff,
            "log_prior": self.log_prior,
            "log_proposal": self.log_proposal,
            "cv_log_pseudo_likelihood": self.cv_log_pseudo_likelihood,
            "log_importance_weight": self.log_importance_weight,
            "posterior_weight": self.posterior_weight,
        }


def aggregate_model_masses(draws: Sequence[ModelDraw]) -> dict[str, Any]:
    """Aggregate posterior mass over families and family-specific choices."""

    if not draws:
        raise ValueError("draws must contain at least one model")
    weights = np.asarray([draw.posterior_weight for draw in draws], dtype=float)
    if np.any(~np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("model posterior weights must be finite and non-negative")
    weights /= weights.sum()

    families = ("knn", "linear", "gaussian")
    family_mass = {family: 0.0 for family in families}
    neighborhood_mass: dict[int, float] = {}
    covariance_mass = {structure: 0.0 for structure in ("isotropic", "diagonal", "full")}
    for draw, weight in zip(draws, weights):
        family_mass[draw.model_family] = family_mass.get(draw.model_family, 0.0) + float(weight)
        if draw.model_family == "knn" and draw.neighborhood_size is not None:
            neighborhood_mass[draw.neighborhood_size] = (
                neighborhood_mass.get(draw.neighborhood_size, 0.0) + float(weight)
            )
        if draw.model_family == "gaussian" and draw.gaussian_covariance_structure is not None:
            covariance_mass[draw.gaussian_covariance_structure] += float(weight)

    def conditional_mass(masses: dict[Any, float], family: str) -> dict[Any, float]:
        total = family_mass[family]
        if total == 0.0:
            return {key: 0.0 for key in masses}
        return {key: value / total for key, value in masses.items()}

    return {
        "model_family": family_mass,
        "by_family": {
            "knn": {
                "neighborhood_size": dict(sorted(neighborhood_mass.items())),
                "neighborhood_size_conditional": dict(
                    sorted(conditional_mass(neighborhood_mass, "knn").items())
                ),
            },
            "linear": {},
            "gaussian": {
                "covariance_structure": covariance_mass,
                "covariance_structure_conditional": conditional_mass(
                    covariance_mass, "gaussian"
                ),
            },
        },
    }
