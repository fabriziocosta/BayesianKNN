"""Internal model-draw records and public serialization."""

from __future__ import annotations

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
    neighborhood_size: int = 0
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
            "neighbourhood_size": self.neighborhood_size,
            "projection_scale_draw": _draw_dict(self.projection_scale_draw),
            "subset_scale_draw": _draw_dict(self.subset_scale_draw),
            "neighbor_scale_draw": _draw_dict(self.neighbor_scale_draw),
            "neighbour_scale_draw": _draw_dict(self.neighbor_scale_draw),
            "projection_beta": self.projection_scale_draw.beta,
            "projection_cutoff": self.projection_scale_draw.cutoff,
            "subset_beta": self.subset_scale_draw.beta,
            "subset_cutoff": self.subset_scale_draw.cutoff,
            "neighbor_beta": self.neighbor_scale_draw.beta,
            "neighbor_cutoff": self.neighbor_scale_draw.cutoff,
            "log_prior": self.log_prior,
            "log_proposal": self.log_proposal,
            "cv_log_pseudo_likelihood": self.cv_log_pseudo_likelihood,
            "log_importance_weight": self.log_importance_weight,
            "posterior_weight": self.posterior_weight,
        }
