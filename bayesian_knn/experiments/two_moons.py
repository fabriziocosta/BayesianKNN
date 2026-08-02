"""Backward-compatible two-moons experiment API."""

from __future__ import annotations

from typing import Any

from .classification_2d import (
    Classification2DResult,
    format_convergence_history,
    plot_probability_heatmap,
    probability_surface,
    run_2d_classification_experiment,
)

TwoMoonsResult = Classification2DResult


def run_two_moons_experiment(
    *,
    n_samples: int = 600,
    noise: float = 0.24,
    data_random_state: int = 7,
    test_size: float = 0.30,
    split_random_state: int = 7,
    model_parameters: dict[str, Any] | None = None,
) -> TwoMoonsResult:
    """Run the generic experiment with the moon dataset selected."""

    return run_2d_classification_experiment(
        "moon",
        dataset_parameters={
            "n_samples": n_samples,
            "noise": noise,
            "random_state": data_random_state,
        },
        test_size=test_size,
        split_random_state=split_random_state,
        model_parameters=model_parameters,
    )


__all__ = [
    "TwoMoonsResult",
    "format_convergence_history",
    "plot_probability_heatmap",
    "probability_surface",
    "run_two_moons_experiment",
]
