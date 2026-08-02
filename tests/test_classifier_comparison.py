import numpy as np
import pytest

from bayesian_knn.experiments.classifier_comparison import (
    format_comparison_table,
    load_standard_dataset,
    plot_comparison_results,
    run_comparison_suite,
    run_dataset_comparison,
)


def test_standard_dataset_loaders_return_numeric_classification_data():
    for dataset in ("iris", "wine", "breast_cancer", "digits"):
        X, y = load_standard_dataset(dataset)
        assert X.ndim == 2
        assert X.shape[0] == y.shape[0]
        assert np.isfinite(X).all()
        assert len(np.unique(y)) >= 2


def test_dataset_comparison_returns_all_classifiers_and_metrics():
    parameters = {
        "representation": "identity",
        "n_estimators": 2,
        "min_subset_size": 10,
        "max_subset_size": 20,
        "cv": 2,
        "n_jobs": 1,
        "random_state": 12,
    }
    result = run_dataset_comparison(
        "iris",
        test_size=0.25,
        random_state=7,
        n_neighbors=3,
        bayesian_parameters=parameters,
        random_forest_parameters={"n_estimators": 5, "n_jobs": 1},
    )
    assert set(result.scores) == {"Bayesian k-NN", "k-NN", "Random forest", "SVM"}
    for metrics in result.scores.values():
        assert set(metrics) == {"accuracy", "balanced_accuracy", "macro_f1"}
        assert all(0.0 <= value <= 1.0 for value in metrics.values())

    suite = run_comparison_suite(
        ("iris",),
        test_size=0.25,
        random_state=7,
        bayesian_parameters=parameters,
        random_forest_parameters={"n_estimators": 5, "n_jobs": 1},
    )
    assert "| Dataset | Classifier |" in format_comparison_table(suite)


def test_comparison_results_can_be_plotted_as_grouped_bars():
    pytest.importorskip("matplotlib")
    parameters = {
        "representation": "identity",
        "n_estimators": 2,
        "min_subset_size": 10,
        "max_subset_size": 20,
        "cv": 2,
        "n_jobs": 1,
        "random_state": 12,
    }
    results = run_comparison_suite(
        ("iris",),
        test_size=0.25,
        random_state=7,
        bayesian_parameters=parameters,
        random_forest_parameters={"n_estimators": 5, "n_jobs": 1},
    )
    figure = plot_comparison_results(results, metric="macro_f1")
    assert len(figure.axes) == 1
