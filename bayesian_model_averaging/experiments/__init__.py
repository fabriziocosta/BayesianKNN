"""Reproducible experiments for BPMA."""

from .classification_2d import (
    Classification2DResult,
    bayes_error_for_dataset,
    format_convergence_history,
    format_error_comparison,
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
    "bayes_error_for_dataset",
    "comparison_results_dataframe",
    "format_convergence_history",
    "format_error_comparison",
    "load_standard_dataset",
    "make_2d_dataset",
    "plot_comparison_results",
    "plot_probability_heatmap",
    "probability_surface",
    "run_comparison_suite",
    "run_dataset_comparison",
    "run_2d_classification_experiment",
]
