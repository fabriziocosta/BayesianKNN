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
    blobs_X, blobs_y = make_2d_dataset("blobs", n_samples=90, n_classes=4)

    assert moon_X.shape == (80, 2)
    assert iris_X.shape == (150, 2)
    assert gaussian_X.shape == (80, 2)
    assert blobs_X.shape == (90, 2)
    assert np.array_equal(np.unique(moon_y), [0, 1])
    assert np.array_equal(np.unique(iris_y), [0, 1, 2])
    assert np.array_equal(np.unique(gaussian_y), [0, 1])
    assert np.array_equal(np.unique(blobs_y), [0, 1, 2, 3])


def test_each_supported_dataset_can_run_through_the_generic_experiment():
    for dataset, parameters in (
        ("moon", {"n_samples": 80}),
        ("iris", {"feature_indices": (2, 3)}),
        ("gaussian", {"n_samples": 80}),
        ("blobs", {"n_samples": 80, "n_classes": 4}),
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


def test_blobs_validate_number_of_classes():
    with pytest.raises(ValueError, match="n_classes"):
        make_2d_dataset("blobs", n_samples=20, n_classes=1)


def test_blobs_expose_center_and_spread_controls():
    X, y = make_2d_dataset(
        "blobs",
        n_samples=900,
        n_classes=3,
        blob_center_radius=2.5,
        blob_cluster_standard_deviation=1.8,
        random_state=4,
    )
    class_means = np.array([X[y == label].mean(axis=0) for label in np.unique(y)])
    assert np.allclose(np.linalg.norm(class_means, axis=1), 2.5, atol=0.25)
    assert np.max(np.linalg.norm(X - class_means[y], axis=1)) > 4.0

    with pytest.raises(ValueError, match="center_radius"):
        make_2d_dataset("blobs", n_classes=3, blob_center_radius=0)
    with pytest.raises(ValueError, match="cluster_standard_deviation"):
        make_2d_dataset("blobs", n_classes=3, blob_cluster_standard_deviation=0)


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
