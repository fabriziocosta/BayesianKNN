"""Reproducible experiments for Bayesian k-NN."""

from .classification_2d import (
    Classification2DResult,
    make_2d_dataset,
    run_2d_classification_experiment,
)
from .classifier_comparison import (
    DatasetComparisonResult,
    comparison_results_dataframe,
    format_comparison_table,
    load_standard_dataset,
    plot_comparison_results,
    run_comparison_suite,
    run_dataset_comparison,
)
from .two_moons import (
    TwoMoonsResult,
    format_convergence_history,
    plot_probability_heatmap,
    run_two_moons_experiment,
)

__all__ = [
    "Classification2DResult",
    "DatasetComparisonResult",
    "comparison_results_dataframe",
    "TwoMoonsResult",
    "format_comparison_table",
    "format_convergence_history",
    "load_standard_dataset",
    "make_2d_dataset",
    "plot_comparison_results",
    "plot_probability_heatmap",
    "run_comparison_suite",
    "run_dataset_comparison",
    "run_two_moons_experiment",
    "run_2d_classification_experiment",
]
