"""Ad hoc runner for the 2D classification notebook."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from time import perf_counter
from typing import Any

from bayesian_model_averaging.experiments.classification_2d import (
    Classification2DResult,
    format_convergence_history,
    format_error_comparison,
    format_family_parameter_shares,
    run_2d_classification_experiment,
)


def run_and_report(
    datasets: Sequence[str],
    *,
    dataset_parameters: Mapping[str, Any],
    split_parameters: Mapping[str, Any],
    model_parameters: Mapping[str, Any],
) -> Iterator[Classification2DResult]:
    """Run datasets, print diagnostics, and yield each result immediately."""

    for dataset in datasets:
        started = perf_counter()
        result = run_2d_classification_experiment(
            dataset,
            dataset_parameters=dict(dataset_parameters),
            **dict(split_parameters),
            model_parameters=dict(model_parameters),
        )
        runtime_seconds = perf_counter() - started
        print(f"dataset: {result.dataset}")
        print(f"runtime: {runtime_seconds:.2f}s")
        print(f"test accuracy: {result.test_accuracy:.3f}")
        print(format_error_comparison(result))
        print(f"estimators used: {result.model.n_estimators_}")
        print(f"converged: {result.model.converged_}")
        family_shares = ", ".join(
            f"{name}={share:.3f}"
            for name, share in result.model_masses["family"].items()
        )
        print(f"family posterior shares: {family_shares}")
        print(
            format_family_parameter_shares(
                result,
                "gaussian_mixture",
                "covariance_structure",
                "  gaussian-mixture covariance structure posterior shares",
            )
        )
        print(
            format_family_parameter_shares(
                result,
                "gaussian_mixture",
                "n_components",
                "  gaussian-mixture component-count posterior shares (top 12)",
                max_items=12,
            )
        )
        print(
            format_family_parameter_shares(
                result,
                "knn",
                "n_neighbors",
                "  knn k posterior shares (top 12)",
                max_items=12,
            )
        )
        print(
            format_family_parameter_shares(
                result,
                "mlp",
                "hidden_layer_sizes",
                "  mlp architecture posterior shares",
            )
        )
        print(
            format_family_parameter_shares(
                result,
                "mlp",
                "activation",
                "  mlp activation posterior shares",
            )
        )
        print(
            format_family_parameter_shares(
                result,
                "decision_tree",
                "max_depth",
                "  decision tree max-depth posterior shares",
            )
        )
        print(
            format_family_parameter_shares(
                result,
                "decision_tree",
                "criterion",
                "  decision tree criterion posterior shares",
            )
        )
        print(format_convergence_history(result))
        yield result
