import numpy as np
import pytest

from bayesian_predictive_model_averaging import (
    CategoricalPrior,
    GaussianCovariancePrior,
    IntegerChoicePrior,
    LogisticLogScalePrior,
    LogisticScalePrior,
    LogUniformPrior,
    SimplicityCategoricalPrior,
)
from bayesian_predictive_model_averaging.priors import make_scale_prior


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


def test_empirical_frequency_matches_the_beta_cutoff_marginal():
    prior = LogisticScalePrior()
    rng = np.random.default_rng(22)
    counts = np.zeros(3, dtype=float)
    expected = np.zeros(3, dtype=float)
    for _ in range(4000):
        draw = prior.draw([1, 2, 3], rng)
        counts[draw.index] += 1
        expected += np.asarray(draw.probabilities)
    assert np.allclose(counts / counts.sum(), expected / expected.sum(), atol=0.025)


def test_scale_prior_accepts_a_logistic_configuration_mapping():
    prior = make_scale_prior({"family": "logistic", "beta_shape": 2.0, "beta_scale": 1.0})
    assert isinstance(prior, LogisticScalePrior)


def test_logistic_log_scale_prior_sweeps_geometric_grid_with_lower_preference():
    prior = LogisticLogScalePrior(low=1e-2, high=1e2, n_values=5)
    assert np.allclose(prior.values, [1e-2, 1e-1, 1.0, 1e1, 1e2])

    draw = prior.draw(np.random.default_rng(14))
    probabilities = np.asarray(draw.probabilities)
    assert draw.value in prior.values
    assert draw.values == prior.values
    assert np.isclose(probabilities.sum(), 1.0)
    assert np.all(probabilities > 0)
    assert np.all(np.diff(probabilities) <= 0)


def test_logistic_log_scale_prior_is_reproducible_and_validates_range():
    prior = LogisticLogScalePrior(low=1e-2, high=1e2, n_values=9, beta_shape=3.0)
    assert prior.draw(np.random.default_rng(10)) == prior.draw(np.random.default_rng(10))
    with pytest.raises(ValueError):
        LogisticLogScalePrior(low=0.0)
    with pytest.raises(ValueError):
        LogisticLogScalePrior(n_values=1)


def test_gaussian_covariance_prior_is_normalized_monotone_and_reproducible():
    prior = GaussianCovariancePrior()
    left = prior.draw(4, np.random.default_rng(31))
    right = prior.draw(4, np.random.default_rng(31))
    probabilities = np.asarray(left.probabilities)

    assert left == right
    assert left.value in {"isotropic", "diagonal", "full"}
    assert np.isclose(probabilities.sum(), 1.0)
    assert np.all(probabilities > 0)
    assert np.all(np.diff(probabilities) < 0)
    assert np.isclose(left.log_probability, np.log(left.probability))


def test_gaussian_covariance_prior_simplicity_controls_complexity_penalty():
    simple = GaussianCovariancePrior(simplicity=0.5).draw(5, np.random.default_rng(1))
    strong = GaussianCovariancePrior(simplicity=3.0).draw(5, np.random.default_rng(1))

    assert strong.probabilities[0] > simple.probabilities[0]
    assert strong.probabilities[-1] < simple.probabilities[-1]


def test_generic_parameter_priors_are_reproducible_and_record_metadata():
    categorical = CategoricalPrior(["relu", "tanh"], [3.0, 1.0])
    left = categorical.draw(np.random.default_rng(4))
    right = categorical.draw(np.random.default_rng(4))
    assert left == right
    assert left[0] in {"relu", "tanh"}
    assert np.isclose(np.exp(left[1]), 0.75 if left[0] == "relu" else 0.25)

    integer = IntegerChoicePrior([1, 2, 3])
    assert integer.draw(np.random.default_rng(2))[0] in {1, 2, 3}

    simplicity = SimplicityCategoricalPrior(["small", "large"], [1, 10])
    assert simplicity.probabilities[0] > simplicity.probabilities[1]


def test_log_uniform_prior_returns_a_finite_density_draw():
    value, log_probability, metadata = LogUniformPrior(1e-4, 1e-1).draw(
        np.random.default_rng(8)
    )
    assert 1e-4 < value < 1e-1
    assert np.isfinite(log_probability)
    assert metadata == {"low": 1e-4, "high": 1e-1}
