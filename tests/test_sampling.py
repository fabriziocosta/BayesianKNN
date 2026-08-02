import numpy as np

from bayesian_knn.sampling import sample_subset


def test_classification_subset_is_cv_admissible_and_probability_is_conditional():
    y = np.repeat([0, 1, 2], 4)
    sample = sample_subset(
        y, size=9, task="classification", n_splits=3, rng=np.random.default_rng(1)
    )
    assert len(sample.indices) == 9
    assert np.all(np.bincount(y[sample.indices], minlength=3) >= 3)
    assert np.isfinite(sample.log_probability)


def test_regression_subset_probability_is_uniform_without_replacement():
    sample = sample_subset(
        np.arange(10), size=4, task="regression", n_splits=2, rng=np.random.default_rng(1)
    )
    assert len(np.unique(sample.indices)) == 4
    assert np.isclose(np.exp(sample.log_probability), 1 / 210)
