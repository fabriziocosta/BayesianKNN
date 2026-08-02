import numpy as np
import pytest

from bayesian_knn.experiments.classification_2d import (
    make_2d_dataset,
    plot_probability_heatmap,
    run_2d_classification_experiment,
)


def _small_model_parameters():
    return {
        "representation": "identity",
        "n_estimators": 2,
        "min_subset_size": 10,
        "max_subset_size": 20,
        "cv": 2,
        "n_jobs": 1,
        "random_state": 12,
    }


def test_supported_2d_dataset_shapes_and_labels():
    moon_X, moon_y = make_2d_dataset("moon", n_samples=80)
    iris_X, iris_y = make_2d_dataset("iris")
    gaussian_X, gaussian_y = make_2d_dataset("gaussian", n_samples=80)

    assert moon_X.shape == (80, 2)
    assert iris_X.shape == (150, 2)
    assert gaussian_X.shape == (80, 2)
    assert np.array_equal(np.unique(moon_y), [0, 1])
    assert np.array_equal(np.unique(iris_y), [0, 1, 2])
    assert np.array_equal(np.unique(gaussian_y), [0, 1])


def test_each_supported_dataset_can_run_through_the_generic_experiment():
    for dataset, parameters in (
        ("moon", {"n_samples": 80}),
        ("iris", {"feature_indices": (2, 3)}),
        ("gaussian", {"n_samples": 80}),
    ):
        result = run_2d_classification_experiment(
            dataset,
            dataset_parameters=parameters,
            test_size=0.25,
            split_random_state=7,
            model_parameters=_small_model_parameters(),
        )
        assert result.X.shape[1] == 2
        assert result.y_pred.shape == result.y_test.shape
        assert 0.0 <= result.test_accuracy <= 1.0
        assert result.probability_class in result.model.classes_


def test_multiclass_plot_creates_one_probability_panel_per_class():
    pytest.importorskip("matplotlib")
    result = run_2d_classification_experiment(
        "iris",
        dataset_parameters={"feature_indices": (2, 3)},
        test_size=0.25,
        model_parameters=_small_model_parameters(),
    )
    figure = plot_probability_heatmap(result, grid_size=12)
    assert len(figure.axes) == 6
