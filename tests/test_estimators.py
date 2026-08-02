import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bayesian_knn import BayesianKNNClassifier, BayesianKNNRegressor, LogisticScalePrior


@pytest.fixture
def data():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(30, 3))
    y_class = np.repeat([0, 1, 2], 10)
    y_reg = X[:, 0] - 0.5 * X[:, 1]
    return X, y_class, y_reg


def estimator_kwargs():
    return dict(
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
    estimator = BayesianKNNClassifier(**estimator_kwargs()).fit(X, y)
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
    estimator = BayesianKNNRegressor(**estimator_kwargs()).fit(X, y)
    prediction = estimator.predict(X[:5])
    assert prediction.shape == (5,)
    assert np.all(np.isfinite(prediction))
    assert np.isfinite(estimator.score(X, y))


def test_parallel_and_serial_draws_are_reproducible(data):
    X, y, _ = data
    serial = BayesianKNNClassifier(**estimator_kwargs()).fit(X, y)
    parallel = BayesianKNNClassifier(**{**estimator_kwargs(), "n_jobs": 2}).fit(X, y)
    assert np.allclose(serial.predict_proba(X), parallel.predict_proba(X))
    assert [draw["subset_indices"].tolist() for draw in serial.get_model_draws()] == [
        draw["subset_indices"].tolist() for draw in parallel.get_model_draws()
    ]


def test_auto_convergence_respects_max_estimators(data):
    X, y, _ = data
    estimator = BayesianKNNClassifier(
        **{**estimator_kwargs(), "n_estimators": "auto", "max_estimators": 2}
    ).fit(X, y)
    assert estimator.n_estimators_ == 2
    assert estimator.converged_ is False


def test_estimator_is_cloneable(data):
    X, y, _ = data
    estimator = BayesianKNNClassifier(**estimator_kwargs())
    cloned = clone(estimator)
    assert cloned.get_params()["random_state"] == 7
    cloned.fit(X, y)


def test_estimator_works_in_pipeline_and_grid_search(data):
    X, y, _ = data
    estimator = BayesianKNNClassifier(
        **{**estimator_kwargs(), "n_estimators": 1, "max_subset_size": 12}
    )
    pipeline = make_pipeline(StandardScaler(), estimator)
    search = GridSearchCV(
        pipeline,
        {"bayesianknnclassifier__max_neighbors": [1, 2]},
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
    BayesianKNNClassifier(
        **{**estimator_kwargs(), "scale_prior": prior, "n_jobs": 1, "n_estimators": 1}
    ).fit(X, y)
    assert prior.calls == 3


def test_regression_scoring_uses_training_fold_variance(data):
    from bayesian_knn.scoring import regression_cv_score

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
