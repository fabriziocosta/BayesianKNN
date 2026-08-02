import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bayesian_model_averaging import (
    BayesianModelAveragingClassifier,
    BayesianModelAveragingRegressor,
    LogisticScalePrior,
)
from bayesian_model_averaging.utils import stable_softmax


@pytest.fixture
def data():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(30, 3))
    y_class = np.repeat([0, 1, 2], 10)
    y_reg = X[:, 0] - 0.5 * X[:, 1]
    return X, y_class, y_reg


def estimator_kwargs():
    return dict(
        model_family="knn",
        representation="identity",
        min_subset_size=9,
        max_subset_size=18,
        cv=3,
        n_estimators=2,
        n_jobs=1,
        random_state=7,
    )


def test_classifier_predicts_and_stores_complete_draws(data):
    X, y, _ = data
    estimator = BayesianModelAveragingClassifier(**estimator_kwargs()).fit(X, y)
    probabilities = estimator.predict_proba(X[:5])
    assert probabilities.shape == (5, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert estimator.predict(X[:5]).shape == (5,)
    assert len(estimator.get_model_draws()) == 2
    assert all(
        draw["neighborhood_size"] <= draw["subset_size"] - int(np.ceil(draw["subset_size"] / 3))
        for draw in estimator.get_model_draws()
    )


def test_regressor_predicts_and_scores(data):
    X, _, y = data
    estimator = BayesianModelAveragingRegressor(**estimator_kwargs()).fit(X, y)
    prediction = estimator.predict(X[:5])
    assert prediction.shape == (5,)
    assert np.all(np.isfinite(prediction))
    assert np.isfinite(estimator.score(X, y))


@pytest.mark.parametrize("representation", ["identity", "gaussian", "sparse"])
def test_classifier_accepts_csr_inputs_for_all_representations(data, representation):
    X, y, _ = data
    X_sparse = csr_matrix(X)
    estimator = BayesianModelAveragingClassifier(
        **{
            **estimator_kwargs(),
            "representation": representation,
        }
    ).fit(X_sparse, y)

    probabilities = estimator.predict_proba(X_sparse[:5])
    assert probabilities.shape == (5, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_regressor_accepts_csr_inputs(data):
    X, _, y = data
    X_sparse = csr_matrix(X)
    estimator = BayesianModelAveragingRegressor(**estimator_kwargs()).fit(X_sparse, y)

    prediction = estimator.predict(X_sparse[:5])
    assert prediction.shape == (5,)
    assert np.all(np.isfinite(prediction))


def test_mixed_representation_samples_identity_and_projection_families(data):
    X, y, _ = data
    estimator = BayesianModelAveragingClassifier(
        **{
            **estimator_kwargs(),
            "representation": "mixed",
            "n_estimators": 20,
        }
    ).fit(X, y)
    draws = estimator.get_model_draws()
    families = {draw["representation_family"] for draw in draws}
    assert families == {"identity", "gaussian", "sparse"}
    assert all(
        draw["representation_family_probability"] == pytest.approx(1 / 3)
        for draw in draws
    )
    identity_draws = [draw for draw in draws if draw["representation_family"] == "identity"]
    assert all(draw["projection_dimension"] == X.shape[1] for draw in identity_draws)


def test_gaussian_family_integrates_covariance_structures(data):
    X, y, _ = data
    estimator = BayesianModelAveragingClassifier(
        **{
            **estimator_kwargs(),
            "model_family": "gaussian",
            "n_estimators": 20,
        }
    ).fit(X, y)
    draws = estimator.get_model_draws()

    assert {draw["model_family"] for draw in draws} == {"gaussian"}
    assert {draw["gaussian_covariance_structure"] for draw in draws} == {
        "isotropic",
        "diagonal",
        "full",
    }
    assert all(draw["gaussian_covariance_draw"] is not None for draw in draws)
    assert all(draw["neighbor_scale_draw"] is None for draw in draws)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_linear_family_is_a_valid_model_family(data, task):
    X, y_class, y_reg = data
    estimator_class = (
        BayesianModelAveragingClassifier
        if task == "classification"
        else BayesianModelAveragingRegressor
    )
    estimator = estimator_class(
        **{**estimator_kwargs(), "model_family": "linear", "n_estimators": 2}
    ).fit(X, y_class if task == "classification" else y_reg)

    draws = estimator.get_model_draws()
    assert all(draw["model_family"] == "linear" for draw in draws)
    assert all(draw["gaussian_covariance_draw"] is None for draw in draws)
    assert all(draw["neighbor_scale_draw"] is None for draw in draws)


def test_mixed_model_family_averages_knn_linear_and_gaussian(data):
    X, y, _ = data
    estimator = BayesianModelAveragingClassifier(
        **{
            **estimator_kwargs(),
            "model_family": "mixed",
            "n_estimators": 20,
        }
    ).fit(X, y)
    draws = estimator.get_model_draws()

    assert {draw["model_family"] for draw in draws} == {"knn", "linear", "gaussian"}
    assert all(draw["model_family_probability"] == pytest.approx(1 / 3) for draw in draws)


def test_model_averaging_defaults_to_mixed_family():
    assert BayesianModelAveragingClassifier().get_params()["model_family"] == "mixed"


def test_temperature_concentrates_mass_on_higher_scoring_models(data):
    X, y, _ = data
    common = {
        **estimator_kwargs(),
        "model_family": "mixed",
        "n_estimators": 20,
    }
    ordinary = BayesianModelAveragingClassifier(**common).fit(X, y)
    sharp = BayesianModelAveragingClassifier(**{**common, "temperature": 0.25}).fit(X, y)

    ordinary_draws = ordinary.get_model_draws()
    sharp_draws = sharp.get_model_draws()
    ordinary_scores = np.asarray(
        [draw["log_importance_weight"] for draw in ordinary_draws]
    )
    sharp_scores = np.asarray([draw["log_importance_weight"] for draw in sharp_draws])
    ordinary_weights = np.asarray(
        [draw["posterior_weight"] for draw in ordinary_draws]
    )
    sharp_weights = np.asarray([draw["posterior_weight"] for draw in sharp_draws])

    assert np.allclose(ordinary_scores, sharp_scores)
    assert np.allclose(ordinary_weights, stable_softmax(ordinary_scores))
    assert np.allclose(sharp_weights, stable_softmax(sharp_scores / 0.25))
    assert np.sum(sharp_weights**2) > np.sum(ordinary_weights**2)


def test_model_masses_report_family_and_nested_choice_mass(data):
    X, y, _ = data
    estimator = BayesianModelAveragingClassifier(
        **{
            **estimator_kwargs(),
            "model_family": "mixed",
            "n_estimators": 20,
        }
    ).fit(X, y)
    masses = estimator.get_model_masses()
    family_mass = masses["model_family"]
    draws = estimator.get_model_draws()
    expected_family_mass = {
        family: sum(
            draw["posterior_weight"]
            for draw in draws
            if draw["model_family"] == family
        )
        for family in ("knn", "linear", "gaussian")
    }

    assert np.isclose(sum(family_mass.values()), 1.0)
    assert family_mass == pytest.approx(expected_family_mass)
    assert masses == estimator.model_masses_
    assert np.isclose(
        sum(masses["by_family"]["knn"]["neighborhood_size"].values()), family_mass["knn"]
    )
    assert np.isclose(
        sum(masses["by_family"]["gaussian"]["covariance_structure"].values()),
        family_mass["gaussian"],
    )
    assert np.isclose(
        sum(masses["by_family"]["knn"]["neighborhood_size_conditional"].values()),
        1.0,
    )
    assert np.isclose(
        sum(masses["by_family"]["gaussian"]["covariance_structure_conditional"].values()),
        1.0,
    )
    assert {draw["model_family"] for draw in draws} == {"knn", "linear", "gaussian"}


def test_parallel_and_serial_draws_are_reproducible(data):
    X, y, _ = data
    serial = BayesianModelAveragingClassifier(**estimator_kwargs()).fit(X, y)
    parallel = BayesianModelAveragingClassifier(**{**estimator_kwargs(), "n_jobs": 2}).fit(X, y)
    assert np.allclose(serial.predict_proba(X), parallel.predict_proba(X))
    assert [draw["subset_indices"].tolist() for draw in serial.get_model_draws()] == [
        draw["subset_indices"].tolist() for draw in parallel.get_model_draws()
    ]


def test_auto_convergence_respects_max_estimators(data):
    X, y, _ = data
    estimator = BayesianModelAveragingClassifier(
        **{**estimator_kwargs(), "n_estimators": "auto", "max_estimators": 2}
    ).fit(X, y)
    assert estimator.n_estimators_ == 2
    assert estimator.converged_ is False


def test_estimator_is_cloneable(data):
    X, y, _ = data
    estimator = BayesianModelAveragingClassifier(**estimator_kwargs())
    cloned = clone(estimator)
    assert cloned.get_params()["random_state"] == 7
    cloned.fit(X, y)


def test_estimator_works_in_pipeline_and_grid_search(data):
    X, y, _ = data
    estimator = BayesianModelAveragingClassifier(
        **{**estimator_kwargs(), "n_estimators": 1, "max_subset_size": 12}
    )
    pipeline = make_pipeline(StandardScaler(), estimator)
    search = GridSearchCV(
        pipeline,
        {"bayesianmodelaveragingclassifier__max_neighbors": [1, 2]},
        cv=2,
        n_jobs=1,
    )
    search.fit(X, y)
    assert search.best_estimator_.predict(X[:2]).shape == (2,)


def test_one_prior_instance_drives_all_three_scale_draws(data):
    X, y, _ = data

    class CountingPrior:
        def __init__(self):
            self.delegate = LogisticScalePrior()
            self.calls = 0

        def draw(self, values, rng):
            self.calls += 1
            return self.delegate.draw(values, rng)

    prior = CountingPrior()
    BayesianModelAveragingClassifier(
        **{**estimator_kwargs(), "scale_prior": prior, "n_jobs": 1, "n_estimators": 1}
    ).fit(X, y)
    assert prior.calls == 3


def test_regression_scoring_uses_training_fold_variance(data):
    from bayesian_model_averaging.scoring import regression_cv_score

    X, _, y = data
    y = y.astype(float)
    train = np.array([0, 1, 2, 3])
    validation = np.array([4, 5])
    score = regression_cv_score(
        X,
        y,
        [(train, validation)],
        n_neighbors=1,
        weights="uniform",
        metric="euclidean",
        epsilon=1e-8,
    )
    from sklearn.neighbors import KNeighborsRegressor

    estimator = KNeighborsRegressor(n_neighbors=1, weights="uniform", metric="euclidean")
    estimator.fit(X[train], y[train])
    residuals = y[validation] - estimator.predict(X[validation])
    sigma2 = np.var(y[train])
    expected = np.mean(-0.5 * (np.log(2 * np.pi * sigma2) + residuals**2 / sigma2))
    assert np.isclose(score, expected)
