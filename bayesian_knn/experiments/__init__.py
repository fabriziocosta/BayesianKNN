"""Reproducible experiments for Bayesian k-NN."""

from .classification_2d import (
    Classification2DResult,
    make_2d_dataset,
    run_2d_classification_experiment,
)
from .two_moons import (
    TwoMoonsResult,
    format_convergence_history,
    plot_probability_heatmap,
    run_two_moons_experiment,
)

__all__ = [
    "Classification2DResult",
    "TwoMoonsResult",
    "format_convergence_history",
    "make_2d_dataset",
    "plot_probability_heatmap",
    "run_two_moons_experiment",
    "run_2d_classification_experiment",
]
