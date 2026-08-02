"""Ad hoc runner for the 2D classification notebook."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from bayesian_model_averaging.experiments.classification_2d import (
    Classification2DResult,
    format_convergence_history,
    format_family_parameter_masses,
    run_2d_classification_experiment,
)


def run_and_report(
    datasets: Sequence[str],
    *,
    dataset_parameters: Mapping[str, Any],
    split_parameters: Mapping[str, Any],
    model_parameters: Mapping[str, Any],
) -> list[Classification2DResult]:
    """Run the configured datasets and print compact diagnostics."""

    results: list[Classification2DResult] = []
    for dataset in datasets:
        result = run_2d_classification_experiment(
            dataset,
            dataset_parameters=dict(dataset_parameters),
            **dict(split_parameters),
            model_parameters=dict(model_parameters),
        )
        print(f"dataset: {result.dataset}")
        print(f"test accuracy: {result.test_accuracy:.3f}")
        print(f"estimators used: {result.model.n_estimators_}")
        print(f"converged: {result.model.converged_}")
        family_masses = ", ".join(
            f"{name}={mass:.3f}"
            for name, mass in result.model_masses["family"].items()
        )
        print(f"family posterior masses: {family_masses}")
        print(
            format_family_parameter_masses(
                result,
                "gaussian",
                "covariance_structure",
                "  gaussian covariance structure masses",
            )
        )
        print(
            format_family_parameter_masses(
                result,
                "knn",
                "n_neighbors",
                "  knn k masses (top 12)",
                max_items=12,
            )
        )
        print(format_convergence_history(result))
        results.append(result)
    return results
