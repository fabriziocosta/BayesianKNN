from pathlib import Path

import numpy as np
import pytest

from bayesian_knn.experiments.two_moons import (
    format_convergence_history,
    plot_probability_heatmap,
    probability_surface,
    run_two_moons_experiment,
)


def test_two_moons_experiment_module_produces_reproducible_outputs(tmp_path):
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
    left = run_two_moons_experiment(model_parameters=parameters)
    right = run_two_moons_experiment(model_parameters=parameters)
    assert left.X_train.shape == (420, 2)
    assert left.X_test.shape == (180, 2)
    assert left.test_accuracy == right.test_accuracy
    assert np.array_equal(left.y_pred, right.y_pred)

    xx, yy, probability = probability_surface(left, grid_size=12)
    assert xx.shape == yy.shape == probability.shape == (12, 12)
    output_path = Path(tmp_path) / "heatmap.png"
    figure = plot_probability_heatmap(left, grid_size=12, output_path=output_path)
    assert output_path.exists()
    assert figure.axes
    assert "20 estimators: initial ensemble" in format_convergence_history(left)
