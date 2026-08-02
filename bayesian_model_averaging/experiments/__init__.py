"""Reproducible experiments for Bayesian model averaging."""

from .classification_2d import (
    Classification2DResult,
    format_convergence_history,
    make_2d_dataset,
    plot_probability_heatmap,
    probability_surface,
    run_2d_classification_experiment,
)
from .classifier_comparison import (
    DatasetComparisonResult,
    comparison_results_dataframe,
    load_standard_dataset,
    plot_comparison_results,
    run_comparison_suite,
    run_dataset_comparison,
)

__all__ = [
    "Classification2DResult",
    "DatasetComparisonResult",
    "comparison_results_dataframe",
    "format_convergence_history",
    "load_standard_dataset",
    "make_2d_dataset",
    "plot_comparison_results",
    "plot_probability_heatmap",
    "probability_surface",
    "run_comparison_suite",
    "run_dataset_comparison",
    "run_2d_classification_experiment",
]
