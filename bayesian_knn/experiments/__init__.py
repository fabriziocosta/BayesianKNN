"""Reproducible experiments for Bayesian k-NN."""

from .two_moons import (
    TwoMoonsResult,
    format_convergence_history,
    plot_probability_heatmap,
    run_two_moons_experiment,
)

__all__ = [
    "TwoMoonsResult",
    "format_convergence_history",
    "plot_probability_heatmap",
    "run_two_moons_experiment",
]
