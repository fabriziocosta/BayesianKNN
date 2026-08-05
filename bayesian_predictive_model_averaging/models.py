"""Internal model-draw records and public serialization."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.special import logsumexp

from .priors import ParameterDraw, ScalePriorDraw
from .utils import stable_softmax


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
        "values": tuple(draw.values),
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
    round_index: int = 0
    proposal_id: str = "prior-0"
    family_proposal_probability: float = 1.0
    generating_log_proposal: float | None = None

    def to_dict(self) -> dict[str, Any]:
        subset_draw = self.subset_scale_draw
        return {
            "family_name": self.family_name,
            "family_prior_probability": self.family_prior_probability,
            "parameters": dict(self.parameters),
            "parameter_prior": self.parameter_prior.to_dict(),
            "round_index": self.round_index,
            "proposal_id": self.proposal_id,
            "family_proposal_probability": self.family_proposal_probability,
            "subset_size": self.subset_size,
            "subset_indices": self.subset_indices.copy(),
            "subset_scale_draw": _scale_draw_dict(subset_draw),
            "log_prior": self.log_prior,
            "log_proposal": self.log_proposal,
            "generating_log_proposal": self.generating_log_proposal,
            "cv_log_pseudo_likelihood": self.cv_log_pseudo_likelihood,
            "log_importance_weight": self.log_importance_weight,
            "posterior_weight": self.posterior_weight,
        }


def aggregate_model_masses(draws: Sequence[ModelDraw]) -> dict[str, Any]:
    """Aggregate normalized predictive shares by family and sampled parameters.

    Parameter shares are conditional within each family, so the values under
    each family/parameter mapping sum to one. This keeps the diagnostic
    independent of any particular adapter's parameter names.
    """

    if not draws:
        raise ValueError("draws must contain at least one model")
    weights = np.asarray([draw.posterior_weight for draw in draws], dtype=float)
    if np.any(~np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("model predictive weights must be finite and non-negative")
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


def recompute_importance_weights(
    draws: Sequence[ModelDraw],
    *,
    target_temperature: float,
    adaptive: bool,
    proposal_history: Sequence[dict[str, float]] = (),
    round_sizes: Sequence[int] = (),
) -> tuple[float, float]:
    """Compute corrected log weights and return ESS and ESS fraction.

    Adaptive proposals currently change only the estimator-family factor. All
    conditional factors remain at their declared prior, so the deterministic
    mixture denominator can be evaluated from round family probabilities.
    """

    if not draws:
        raise ValueError("draws must contain at least one model")
    if not np.isfinite(target_temperature) or target_temperature <= 0:
        raise ValueError("target_temperature must be finite and positive")

    if adaptive:
        if len(proposal_history) != len(round_sizes) or not proposal_history:
            raise ValueError("adaptive weighting requires proposal history and round sizes")
        if any(size < 1 for size in round_sizes) or sum(round_sizes) != len(draws):
            raise ValueError("round sizes must partition the complete draw population")
        log_round_mass = np.log(np.asarray(round_sizes, dtype=float) / sum(round_sizes))

    log_weights = []
    for draw in draws:
        if adaptive:
            conditional_log_prior = draw.log_prior - np.log(draw.family_prior_probability)
            family_terms = [
                log_round_mass[round_index] + np.log(proposal[draw.family_name])
                for round_index, proposal in enumerate(proposal_history)
            ]
            draw.log_proposal = float(conditional_log_prior + logsumexp(family_terms))
        elif draw.generating_log_proposal is not None:
            draw.log_proposal = float(draw.generating_log_proposal)

        if not adaptive and draw.generating_log_proposal is not None:
            # With q(theta) == p(theta), preserve the existing score-only
            # behavior exactly while retaining the general formula above for
            # adaptive proposals.
            draw.log_importance_weight = float(
                draw.cv_log_pseudo_likelihood
            )
            log_weights.append(
                draw.cv_log_pseudo_likelihood / target_temperature
            )
        else:
            log_target = draw.log_prior + draw.cv_log_pseudo_likelihood / target_temperature
            draw.log_importance_weight = float(log_target - draw.log_proposal)
            log_weights.append(draw.log_importance_weight)

    normalized = stable_softmax(np.asarray(log_weights, dtype=float))
    for draw, weight in zip(draws, normalized):
        draw.posterior_weight = float(weight)
    ess = float(1.0 / np.sum(normalized**2))
    return ess, float(ess / len(draws))
