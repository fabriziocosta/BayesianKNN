import numpy as np
import pytest

from bayesian_knn import LogisticScalePrior


def test_logistic_scale_prior_is_normalized_and_monotone():
    draw = LogisticScalePrior().draw([1, 2, 3, 4], np.random.default_rng(4))
    probabilities = np.asarray(draw.probabilities)
    assert np.isclose(probabilities.sum(), 1.0)
    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities > 0)
    assert np.all(np.diff(probabilities) <= 0)
    assert np.isclose(draw.log_probability, np.log(draw.probability))


def test_logistic_scale_prior_is_reproducible():
    prior = LogisticScalePrior(beta_shape=2.0, beta_scale=1.0)
    left = prior.draw([2, 4, 8], np.random.default_rng(10))
    right = prior.draw([2, 4, 8], np.random.default_rng(10))
    assert left == right


def test_single_value_still_samples_latent_parameters():
    draw = LogisticScalePrior().draw([7], np.random.default_rng(2))
    assert draw.value == 7
    assert draw.probability == 1.0
    assert draw.log_probability == 0.0
    assert draw.probabilities == (1.0,)


def test_prior_rejects_invalid_values():
    with pytest.raises(ValueError):
        LogisticScalePrior().draw([], np.random.default_rng(1))
    with pytest.raises(ValueError):
        LogisticScalePrior().draw([1, 1], np.random.default_rng(1))
    with pytest.raises(ValueError):
        LogisticScalePrior(beta_shape=0)


def test_smaller_values_are_sampled_more_frequently():
    prior = LogisticScalePrior()
    rng = np.random.default_rng(12)
    counts = np.zeros(4, dtype=int)
    for _ in range(1500):
        counts[prior.draw([1, 2, 3, 4], rng).index] += 1
    assert counts[0] > counts[-1]
