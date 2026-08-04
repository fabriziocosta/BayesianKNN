import json

import numpy as np
from sklearn.base import BaseEstimator

from bayesian_predictive_model_averaging import (
    BayesianPredictiveModelAveragingClassifier,
    FamilyRegistration,
    KNNAdapter,
)
from bayesian_predictive_model_averaging.models import ModelDraw, recompute_importance_weights
from bayesian_predictive_model_averaging.priors import ParameterDraw


def make_draw(
    family: str,
    family_prior: float,
    score: float,
    *,
    round_index: int = 0,
    family_proposal: float | None = None,
    conditional_prior: float = 1.0,
) -> ModelDraw:
    log_prior = float(np.log(family_prior * conditional_prior))
    proposal = family_proposal if family_proposal is not None else family_prior
    log_proposal = float(np.log(proposal * conditional_prior))
    return ModelDraw(
        family_name=family,
        family_prior_probability=family_prior,
        parameters={},
        parameter_prior=ParameterDraw({}, 0.0, {}),
        round_index=round_index,
        proposal_id=f"round-{round_index}",
        family_proposal_probability=proposal,
        log_prior=log_prior,
        log_proposal=log_proposal,
        generating_log_proposal=log_proposal,
        cv_log_pseudo_likelihood=score,
    )


def test_prior_proposal_reduces_to_score_only_weights():
    draws = [make_draw("a", 0.5, 0.0), make_draw("b", 0.5, 1.0)]
    recompute_importance_weights(
        draws,
        target_temperature=1.0,
        adaptive=True,
        proposal_history=[{"a": 0.5, "b": 0.5}],
        round_sizes=[2],
    )
    assert np.allclose(
        [draw.log_importance_weight for draw in draws],
        [0.0, 1.0],
    )
    assert np.allclose(
        [draw.posterior_weight for draw in draws],
        np.exp([-0.0, -1.0]) / np.exp([-0.0, -1.0]).sum(),
    )


def test_deterministic_mixture_weight_is_exact():
    draws = [
        make_draw("a", 0.5, 1.0, round_index=1, family_proposal=0.8, conditional_prior=0.25),
        make_draw("b", 0.5, 0.0, round_index=0, family_proposal=0.5, conditional_prior=0.25),
    ]
    recompute_importance_weights(
        draws,
        target_temperature=1.0,
        adaptive=True,
        proposal_history=[{"a": 0.5, "b": 0.5}, {"a": 0.8, "b": 0.2}],
        round_sizes=[1, 1],
    )
    expected = np.log(0.5 / 0.65) + 1.0
    np.testing.assert_allclose(draws[0].log_importance_weight, expected)
    np.testing.assert_allclose(draws[0].log_proposal, np.log(0.65 * 0.25))


def test_family_proposal_is_defensive_and_adapts_toward_predictive_mass():
    class NamedAdapter(BaseEstimator):
        def __init__(self, name):
            self.name = name
            self.supported_tasks = frozenset({"classification"})

    model = BayesianPredictiveModelAveragingClassifier(
        adaptive_importance_sampling=True,
        defensive_prior_weight=0.2,
    )
    model.family_registry_ = (
        FamilyRegistration(NamedAdapter("a"), 0.5),
        FamilyRegistration(NamedAdapter("b"), 0.5),
    )
    model._prior_family_probabilities_ = {"a": 0.5, "b": 0.5}
    model.model_masses_ = {"family": {"a": 0.99, "b": 0.01}}
    proposal = model._next_family_proposal()
    assert proposal["a"] > proposal["b"] > 0
    assert np.isclose(sum(proposal.values()), 1.0)
    assert proposal["b"] >= 0.2 * 0.5


def test_adaptive_fit_records_round_diagnostics_and_stops_at_budget():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(24, 3))
    y = np.repeat([0, 1], 12)
    estimator = BayesianPredictiveModelAveragingClassifier(
        family_registry=[FamilyRegistration(KNNAdapter(), 1.0)],
        adaptive_importance_sampling=True,
        round_size=2,
        min_rounds=2,
        stopping_patience=1,
        max_estimators=4,
        max_rounds=10,
        min_subset_size=8,
        max_subset_size=12,
        cv=2,
        n_jobs=1,
        random_state=4,
    ).fit(X, y)
    assert estimator.n_estimators_ <= 4
    assert estimator.n_rounds_ >= 2
    assert estimator.stopping_reason_ in {"converged", "max_estimators", "max_rounds"}
    assert np.isclose(sum(estimator.proposal_history_[0].values()), 1.0)
    assert np.isclose(sum(draw["posterior_weight"] for draw in estimator.get_model_draws()), 1.0)
    json.dumps(estimator.round_history_)
    assert all("effective_sample_size" in entry for entry in estimator.round_history_)


def test_adaptive_estimator_is_cloneable():
    estimator = BayesianPredictiveModelAveragingClassifier(
        adaptive_importance_sampling=True,
        round_size=4,
        max_estimators=8,
        adaptation_temperature=2.0,
    )
    params = estimator.get_params()
    assert params["adaptive_importance_sampling"] is True
    assert params["round_size"] == 4
    assert params["adaptation_temperature"] == 2.0


def test_adaptive_fit_is_reproducible_across_parallelism():
    rng = np.random.default_rng(14)
    X = rng.normal(size=(24, 3))
    y = np.repeat([0, 1], 12)
    common = dict(
        family_registry=[FamilyRegistration(KNNAdapter(), 1.0)],
        adaptive_importance_sampling=True,
        round_size=2,
        min_rounds=2,
        stopping_patience=1,
        max_estimators=4,
        min_subset_size=8,
        max_subset_size=12,
        cv=2,
        random_state=8,
    )
    serial = BayesianPredictiveModelAveragingClassifier(**common, n_jobs=1).fit(X, y)
    parallel = BayesianPredictiveModelAveragingClassifier(**common, n_jobs=2).fit(X, y)
    assert np.allclose(serial.predict_proba(X), parallel.predict_proba(X))
    assert serial.round_history_ == parallel.round_history_
    assert [draw["log_importance_weight"] for draw in serial.get_model_draws()] == [
        draw["log_importance_weight"] for draw in parallel.get_model_draws()
    ]
