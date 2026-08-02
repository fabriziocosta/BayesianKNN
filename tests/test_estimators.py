import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from bayesian_model_averaging import (
    BayesianModelAveragingClassifier,
    BayesianModelAveragingRegressor,
    FamilyRegistration,
    GaussianAdapter,
    KNNAdapter,
    LinearAdapter,
    LogisticScalePrior,
    MLPAdapter,
    ParameterDraw,
    SamplingContext,
)
from bayesian_model_averaging.scoring import regression_cv_score
from bayesian_model_averaging.utils import stable_softmax


@pytest.fixture
def data():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(30, 3))
    y_class = np.repeat([0, 1, 2], 10)
    y_reg = X[:, 0] - 0.5 * X[:, 1]
    return X, y_class, y_reg


def registration(adapter, weight=1.0):
    return [FamilyRegistration(adapter, weight)]


def estimator_kwargs(adapter=None):
    return dict(
        family_registry=registration(adapter or KNNAdapter()),
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
    draws = estimator.get_model_draws()
    assert len(draws) == 2
    assert all(draw["family_name"] == "knn" for draw in draws)
    assert all(
        draw["parameters"]["n_neighbors"]
        <= draw["subset_size"] - int(np.ceil(draw["subset_size"] / 3))
        for draw in draws
    )


def test_builtin_classifier_concentrations_are_comparable():
    parameters = {"n_neighbors": 8}
    assert KNNAdapter().predictive_concentration("classification", parameters) == 1.0
    assert LinearAdapter().predictive_concentration("classification", {}) == 1.0
    assert GaussianAdapter().predictive_concentration("classification", {}) == 1.0
    assert MLPAdapter().predictive_concentration("classification", {}) == 1.0


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
    estimator = BayesianModelAveragingClassifier(
        **{
            **estimator_kwargs(),
            "representation": representation,
        }
    ).fit(csr_matrix(X), y)
    probabilities = estimator.predict_proba(csr_matrix(X[:5]))
    assert probabilities.shape == (5, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_regressor_accepts_csr_inputs(data):
    X, _, y = data
    estimator = BayesianModelAveragingRegressor(**estimator_kwargs()).fit(csr_matrix(X), y)
    prediction = estimator.predict(csr_matrix(X[:5]))
    assert prediction.shape == (5,)
    assert np.all(np.isfinite(prediction))


def test_mixed_representation_samples_identity_and_projection_families(data):
    X, y, _ = data
    estimator = BayesianModelAveragingClassifier(
        **{
            **estimator_kwargs(),
            "family_registry": registration(KNNAdapter()),
            "representation": "mixed",
            "n_estimators": 20,
        }
    ).fit(X, y)
    draws = estimator.get_model_draws()
    families = {draw["representation_family"] for draw in draws}
    assert families == {"identity", "gaussian", "sparse"}
    identity_draws = [draw for draw in draws if draw["representation_family"] == "identity"]
    assert all(draw["projection_dimension"] == X.shape[1] for draw in identity_draws)


@pytest.mark.parametrize("adapter_class", [LinearAdapter, GaussianAdapter])
@pytest.mark.parametrize("representation", ["identity", "gaussian", "sparse"])
def test_linear_and_gaussian_adapters_accept_all_representations(
    data, adapter_class, representation
):
    X, y, _ = data
    estimator = BayesianModelAveragingClassifier(
        **{
            **estimator_kwargs(adapter_class()),
            "representation": representation,
            "n_estimators": 1,
        }
    ).fit(X, y)
    assert estimator.get_model_draws()[0]["representation_family"] == representation


def test_gaussian_adapter_integrates_covariance_structures(data):
    X, y, _ = data
    estimator = BayesianModelAveragingClassifier(
        **{
            **estimator_kwargs(GaussianAdapter()),
            "n_estimators": 20,
        }
    ).fit(X, y)
    draws = estimator.get_model_draws()
    assert {draw["parameters"]["covariance_structure"] for draw in draws} == {
        "isotropic",
        "diagonal",
        "full",
    }
    assert all(draw["parameter_prior"]["metadata"]["covariance_draw"] for draw in draws)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_linear_adapter_is_valid_for_both_tasks(data, task):
    X, y_class, y_reg = data
    estimator_class = (
        BayesianModelAveragingClassifier
        if task == "classification"
        else BayesianModelAveragingRegressor
    )
    estimator = estimator_class(
        **{**estimator_kwargs(LinearAdapter()), "n_estimators": 2}
    ).fit(X, y_class if task == "classification" else y_reg)
    assert all(draw["family_name"] == "linear" for draw in estimator.get_model_draws())


def test_default_registry_contains_built_in_families(data):
    X, y, _ = data
    estimator = BayesianModelAveragingClassifier(
        representation="identity",
        min_subset_size=9,
        max_subset_size=18,
        cv=3,
        n_estimators=20,
        n_jobs=1,
        random_state=7,
    ).fit(X, y)
    names = {draw["family_name"] for draw in estimator.get_model_draws()}
    assert names == {"knn", "linear", "gaussian"}
    assert all(
        draw["family_prior_probability"] == pytest.approx(1 / 3)
        for draw in estimator.get_model_draws()
    )
    masses = estimator.get_model_masses()
    assert np.isclose(sum(masses["family"].values()), 1.0)
    assert np.isclose(
        sum(masses["parameter"]["gaussian"]["covariance_structure"].values()),
        1.0,
    )
    assert np.isclose(sum(masses["parameter"]["knn"]["n_neighbors"].values()), 1.0)


def test_explicit_family_weights_are_normalized(data):
    X, y, _ = data
    estimator = BayesianModelAveragingClassifier(
        **{
            **estimator_kwargs(LinearAdapter()),
            "family_registry": [
                FamilyRegistration(LinearAdapter(), 1.0),
                FamilyRegistration(KNNAdapter(), 3.0),
            ],
            "n_estimators": 20,
        }
    ).fit(X, y)
    probabilities = {
        draw["family_name"]: draw["family_prior_probability"]
        for draw in estimator.get_model_draws()
    }
    assert probabilities == {"linear": pytest.approx(0.25), "knn": pytest.approx(0.75)}


def test_mlp_adapter_samples_structured_parameters(data):
    X, y, _ = data
    estimator = BayesianModelAveragingClassifier(
        **{**estimator_kwargs(MLPAdapter()), "n_estimators": 3}
    ).fit(X, y)
    draws = estimator.get_model_draws()
    assert all(draw["family_name"] == "mlp" for draw in draws)
    assert all(draw["parameters"]["activation"] in {"relu", "tanh", "logistic"} for draw in draws)
    assert all(isinstance(draw["parameters"]["hidden_layer_sizes"], tuple) for draw in draws)
    assert all(np.isfinite(draw["parameter_prior"]["log_probability"]) for draw in draws)


def test_mlp_adapter_supports_regression(data):
    X, _, y = data
    estimator = BayesianModelAveragingRegressor(
        **{**estimator_kwargs(MLPAdapter()), "n_estimators": 1}
    ).fit(X, y)
    assert estimator.predict(X[:3]).shape == (3,)


class ToyClassifierAdapter(BaseEstimator):
    name = "toy"
    supported_tasks = frozenset({"classification"})
    supported_representations = frozenset({"identity"})

    def sample_parameters(
        self, context: SamplingContext, rng: np.random.Generator
    ) -> ParameterDraw:
        depth = int(rng.choice([1, 2]))
        return ParameterDraw(
            parameters={"max_depth": depth},
            log_probability=-np.log(2.0),
            metadata={"max_depth": {"value": depth, "probability": 0.5}},
        )

    def build_estimator(self, task, parameters, random_state):
        return DecisionTreeClassifier(max_depth=parameters["max_depth"], random_state=random_state)

    def predictive_concentration(self, task, parameters):
        return 1.0


class ToyRegressorAdapter(BaseEstimator):
    name = "toy-regressor"
    supported_tasks = frozenset({"regression"})
    supported_representations = frozenset({"identity"})

    def sample_parameters(
        self, context: SamplingContext, rng: np.random.Generator
    ) -> ParameterDraw:
        return ParameterDraw(parameters={"alpha": 1.0}, log_probability=0.0, metadata={})

    def build_estimator(self, task, parameters, random_state):
        return Ridge(alpha=parameters["alpha"])

    def predictive_concentration(self, task, parameters):
        return 1.0


def test_custom_classifier_adapter_registers_without_core_changes(data):
    X, y, _ = data
    estimator = BayesianModelAveragingClassifier(
        **{**estimator_kwargs(ToyClassifierAdapter()), "n_estimators": 2}
    ).fit(X, y)
    assert {draw["family_name"] for draw in estimator.get_model_draws()} == {"toy"}
    assert estimator.predict_proba(X[:3]).shape == (3, 3)


def test_custom_regressor_adapter_registers_without_core_changes(data):
    X, _, y = data
    estimator = BayesianModelAveragingRegressor(
        **{**estimator_kwargs(ToyRegressorAdapter()), "n_estimators": 2}
    ).fit(X, y)
    assert {draw["family_name"] for draw in estimator.get_model_draws()} == {"toy-regressor"}
    assert estimator.predict(X[:3]).shape == (3,)


def test_adapter_validation_rejects_wrong_task(data):
    X, y, _ = data
    with pytest.raises(ValueError, match="no registered adapter supports task"):
        BayesianModelAveragingRegressor(
            **{**estimator_kwargs(ToyClassifierAdapter()), "n_estimators": 1}
        ).fit(X, y)


def test_temperature_concentrates_mass_on_higher_scoring_models(data):
    X, y, _ = data
    common = {
        **estimator_kwargs(),
        "n_estimators": 20,
    }
    ordinary = BayesianModelAveragingClassifier(**common).fit(X, y)
    sharp = BayesianModelAveragingClassifier(**{**common, "temperature": 0.25}).fit(X, y)
    ordinary_draws = ordinary.get_model_draws()
    sharp_draws = sharp.get_model_draws()
    ordinary_scores = np.asarray([draw["log_importance_weight"] for draw in ordinary_draws])
    sharp_scores = np.asarray([draw["log_importance_weight"] for draw in sharp_draws])
    ordinary_weights = np.asarray([draw["posterior_weight"] for draw in ordinary_draws])
    sharp_weights = np.asarray([draw["posterior_weight"] for draw in sharp_draws])
    assert np.allclose(ordinary_scores, sharp_scores)
    assert np.allclose(ordinary_weights, stable_softmax(ordinary_scores))
    assert np.allclose(sharp_weights, stable_softmax(sharp_scores / 0.25))
    assert np.sum(sharp_weights**2) > np.sum(ordinary_weights**2)


def test_parallel_and_serial_draws_are_reproducible(data):
    X, y, _ = data
    serial = BayesianModelAveragingClassifier(**estimator_kwargs()).fit(X, y)
    parallel = BayesianModelAveragingClassifier(
        **{**estimator_kwargs(), "n_jobs": 2}
    ).fit(X, y)
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


def test_estimator_is_cloneable_and_works_in_grid_search(data):
    X, y, _ = data
    estimator = BayesianModelAveragingClassifier(**estimator_kwargs())
    cloned = clone(estimator)
    assert cloned.get_params()["family_registry"][0].adapter.name == "knn"
    search = GridSearchCV(
        make_pipeline(
            StandardScaler(),
            BayesianModelAveragingClassifier(
                **{**estimator_kwargs(), "max_subset_size": 12}
            ),
        ),
        {
            "bayesianmodelaveragingclassifier__family_registry": [
                registration(KNNAdapter(max_neighbors=1)),
                registration(KNNAdapter(max_neighbors=2)),
            ]
        },
        cv=2,
        n_jobs=1,
    )
    search.fit(X, y)
    assert search.best_estimator_.predict(X[:2]).shape == (2,)


def test_one_prior_instance_drives_projection_subset_and_knn_draws(data):
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
    X, _, y = data
    y = y.astype(float)
    train = np.array([0, 1, 2, 3])
    validation = np.array([4, 5])
    adapter = KNNAdapter()
    parameters = {"n_neighbors": 1, "weights": "uniform", "metric": "euclidean"}
    score = regression_cv_score(
        X,
        y,
        [(train, validation)],
        adapter,
        parameters,
        epsilon=1e-8,
        seed=7,
    )
    estimator = adapter.build_estimator("regression", parameters, 7)
    estimator.fit(X[train], y[train])
    residuals = y[validation] - estimator.predict(X[validation])
    sigma2 = np.var(y[train])
    expected = np.mean(-0.5 * (np.log(2 * np.pi * sigma2) + residuals**2 / sigma2))
    assert np.isclose(score, expected)
