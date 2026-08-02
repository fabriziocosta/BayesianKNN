# Bayesian Model Averaging Architecture

This document describes the implemented architecture. The package is a
scikit-learn-compatible ensemble that samples complete predictive models from
declared priors, scores them with cross-validation, and averages their
predictions. It does not optimize one hyperparameter configuration.

## Design

The core integrates four kinds of uncertainty:

1. estimator family;
2. feature representation and projection dimension;
3. training subset;
4. estimator-family parameters.

Each Monte Carlo draw is a complete, fitted model. The draw records its prior
and cross-validated score, while the ensemble weight is determined by the
cross-validated pseudo-likelihood. This keeps the integration engine generic:
family-specific parameter logic lives in adapters, and scoring only depends on
the estimator contract.

The fit pipeline is:

```text
validate data and CV
        |
normalize runtime family registry
        |
sample family -> compatible representation -> projection parameters
        |
sample CV-admissible subset -> construct SamplingContext
        |
adapter samples complete parameters
        |
generic CV score -> fit final estimator -> store ModelDraw
        |
stable softmax over CV scores -> predictions, posterior shares, convergence
```

## Package layout

```text
bayesian_model_averaging/
  __init__.py          public exports
  base.py              shared sklearn estimator and ensemble engine
  classifier.py        BayesianModelAveragingClassifier
  regressor.py         BayesianModelAveragingRegressor
  adapters.py          adapter protocol, registrations, built-in adapters
  priors.py            reusable prior objects and draw records
  sampling.py          complete-model sampling, CV subsets, fitting
  scoring.py           generic classification and regression scoring
  models.py            ModelDraw and posterior-share diagnostics
  model_families.py    GaussianClassifier implementation
  convergence.py       prediction-change convergence metrics
  utils.py             deterministic seeds and numerical helpers
  representation/
    base.py            representation interface and factory
    identity.py        unchanged row features
    gaussian_projection.py
    sparse_projection.py
  experiments/
    classification_2d.py
```

## Estimator API

The public estimators are:

```python
from bayesian_model_averaging import (
    BayesianModelAveragingClassifier,
    BayesianModelAveragingRegressor,
)

classifier = BayesianModelAveragingClassifier(
    n_estimators=40,
    random_state=7,
)
classifier.fit(X_train, y_train)
probabilities = classifier.predict_proba(X_test)
predictions = classifier.predict(X_test)

regressor = BayesianModelAveragingRegressor(n_estimators=40, random_state=7)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)
```

Both estimators expose `fit`, `predict`, `score`, and `get_model_draws`.
Classifiers additionally expose `predict_proba`; regressors intentionally do
not. The estimators preserve constructor parameters through sklearn cloning,
pipelines, and `GridSearchCV`.

Important constructor parameters are:

- `family_registry`: runtime estimator-family registrations;
- `representation`: `"identity"`, `"gaussian"`, `"sparse"`, or `"mixed"`;
- `scale_prior`: the prior used for projection dimension and subset size;
- `min_subset_size`, `max_subset_size`, and `cv`;
- `n_estimators`, or `"auto"` with `max_estimators`;
- `tolerance`, `convergence_metric`, and `convergence_size`;
- `alpha` for classification probability smoothing;
- `epsilon` for the regression variance floor;
- `temperature` for ensemble-weight concentration;
- `n_jobs` and `random_state`.

## Runtime family adapters

Family selection is registry-driven. The core does not branch on family names.
The public interfaces are:

```python
class EstimatorFamilyAdapter(Protocol):
    name: str
    supported_tasks: frozenset[str]
    supported_representations: frozenset[str]

    def sample_parameters(context, rng) -> ParameterDraw: ...

    def build_estimator(task, parameters, random_state) -> BaseEstimator: ...

    def predictive_concentration(task, parameters) -> float: ...

@dataclass(frozen=True)
class FamilyRegistration:
    adapter: EstimatorFamilyAdapter
    prior_weight: float = 1.0
```

`FamilyRegistration` entries may also be supplied as adapters directly. The
registry normalizes positive weights to sum to one, rejects duplicate names,
and filters families that do not support the requested task. A one-entry
registry is therefore a fixed-family ensemble. If `family_registry` is not
specified, the default is a uniform mixture of k-NN, linear, and Gaussian
families. MLP is available but opt-in.

### Adapter responsibilities

An adapter owns all family-specific parameter logic. `sample_parameters` gets
a `SamplingContext` containing task, transformed feature count, class count,
subset size, minimum CV training-fold size, classes, and the shared scale
prior. It must return one `ParameterDraw` containing:

```python
ParameterDraw(
    parameters={...},       # complete valid estimator configuration
    log_probability=...,    # joint log prior of that configuration
    metadata={...},         # conditional-prior diagnostics
)
```

The core never interprets individual parameter names. This supports conditional
and structured priors, including neural-network architecture choices.

`build_estimator` must return a fresh sklearn-compatible estimator. For
classification, the built estimator must provide `predict_proba`; adapters that
only expose `decision_function` are rejected in this version. An adapter can
override `predictive_concentration`, but the value must be positive and finite.

### Built-in adapters

| Adapter | Classification | Regression | Representations | Main parameters |
| --- | --- | --- | --- | --- |
| `KNNAdapter` | `KNeighborsClassifier` | `KNeighborsRegressor` | identity, Gaussian projection, sparse projection | `n_neighbors`, `weights`, `metric` |
| `LinearAdapter` | `LogisticRegression` | `Ridge` | identity, Gaussian projection, sparse projection | solver/iterations or ridge `alpha` |
| `GaussianAdapter` | `GaussianClassifier` | `BayesianRidge` | identity, Gaussian projection, sparse projection | classification covariance structure |
| `MLPAdapter` | `MLPClassifier` | `MLPRegressor` | identity | architecture, activation, regularization, learning rate |

The Gaussian classifier fits a separate Gaussian likelihood for every class.
Its sampled covariance structure is one of `isotropic`, `diagonal`, or `full`.
Gaussian regression uses `BayesianRidge`; covariance structure is not sampled
for regression.

All built-in adapters currently return predictive concentration `1.0`. In
particular, k-NN neighbourhood size is already expressed in the k-NN
probability vector and is not applied a second time as a confidence multiplier.
This calibration choice keeps family CV scores comparable.

### Registering a new family

A new sklearn family requires one adapter and its priors; the core, scorer,
model record, and estimator classes do not need modification.

```python
class ToyAdapter:
    name = "toy"
    supported_tasks = frozenset({"classification"})
    supported_representations = frozenset({"identity"})

    def sample_parameters(self, context, rng):
        value, log_probability, metadata = my_prior.draw(rng)
        return ParameterDraw(
            parameters={"my_parameter": value},
            log_probability=log_probability,
            metadata=metadata,
        )

    def build_estimator(self, task, parameters, random_state):
        return MySklearnClassifier(my_parameter=parameters["my_parameter"])

    def predictive_concentration(self, task, parameters):
        return 1.0

model = BayesianModelAveragingClassifier(
    family_registry=[FamilyRegistration(ToyAdapter(), prior_weight=1.0)]
)
```

## Representations

The representation is sampled independently for each model draw, subject to
the selected adapter's `supported_representations`.

- `identity` passes the row features through unchanged.
- `gaussian` uses a Gaussian random projection.
- `sparse` uses a sparse random projection.
- `mixed` samples uniformly from the compatible representation families for
  that adapter.

There is no separate `auto` spelling in the current API; `mixed` is the mode
that integrates over identity, Gaussian, and sparse representations where the
adapter permits them. Thus `representation="identity"` always uses the input
row features, while `representation="mixed"` can use any compatible option on
different draws.

Projection dimensions are sampled from the shared `LogisticScalePrior` over
`1..n_features`. Identity uses the singleton dimension `[n_features]`, so it
never truncates the feature space. The representation object is fitted on the
sampled training data and retained with the model draw for prediction.

Linear and Gaussian adapters are representation-agnostic in this design: they
consume whichever transformed feature matrix the core supplies. MLP remains
identity-only until its projected-feature behavior is explicitly validated.

## Priors

`priors.py` contains reusable prior objects. Each prior returns both a sampled
value and the log probability of that value, together with metadata sufficient
for diagnostics.

### Ordered scale prior

`LogisticScalePrior` is shared by projection dimension and subset size, and is
available to adapters for ordered parameters such as k-NN neighbourhood size.
For ordered values with normalized positions `u`, each draw samples:

```text
beta   ~ Gamma(beta_shape, beta_scale)
cutoff ~ Uniform(0, 1)
q(u)   = 1 / (1 + exp(beta * (u - cutoff)))
p(u)   = q(u) / sum(q)
```

The implementation evaluates the distribution in log space, validates a
strictly increasing non-empty integer sequence, and records the full discrete
probability vector in `ScalePriorDraw`.

### Generic parameter priors

- `CategoricalPrior` samples finite values.
- `IntegerChoicePrior` validates finite integer choices.
- `LogUniformPrior` samples positive continuous values on a log scale.
- `SimplicityCategoricalPrior` decreases mass with declared complexity.
- `GaussianCovariancePrior` weights covariance structures by the number of
  covariance parameters:

  ```text
  isotropic: 1
  diagonal:  n_features
  full:      n_features * (n_features + 1) / 2
  ```

`MLPAdapter` uses a finite, simplicity-weighted architecture prior rather than
independently sampling incompatible layer fields. Its joint parameter prior
covers `hidden_layer_sizes`, activation (including `logistic`), `alpha`, and
`learning_rate_init`.

## Sampling a complete model

`sampling.py` performs the following sequence for every deterministic child
seed:

1. select a registered family using normalized family weights;
2. validate the selected task and choose a compatible representation;
3. sample projection dimension and fit the representation;
4. sample an admissible subset size and a subset uniformly conditional on that
   size;
5. construct the CV splitter and `SamplingContext`;
6. ask the adapter for a complete parameter draw;
7. score the configuration through the generic scorer;
8. build and fit the final estimator on the transformed subset;
9. store the resulting `ModelDraw`.

Classification subsets must contain at least `cv` observations from every
global class. Regression subsets must contain at least `cv` observations.
The sampled k-NN neighbourhood size is bounded by the smallest CV training
fold, and invalid configurations fail rather than being silently scored.

The stored prior contribution is:

```text
log_prior =
    log(family prior)
  + log(representation prior)
  + projection-scale log probability
  + subset-size log probability
  + conditional subset log probability
  + adapter-parameter log probability
```

`log_proposal` is equal to `log_prior` because the current sampler proposes
directly from the declared prior. Projection-matrix density terms are not
added because the matrix is sampled from the representation's prior.

## Generic scoring

### Classification

`classification_cv_score` works with any adapter that builds an estimator with
`predict_proba`:

1. fit a fresh estimator on each training fold;
2. align local estimator classes to the global class ordering;
3. apply predictive concentration and Dirichlet smoothing;
4. average validation log probabilities.

For class probability `p`, concentration `kappa`, and smoothing `alpha`:

```text
p_smoothed = (kappa * p + alpha) /
             (kappa + number_of_global_classes * alpha)
```

Absent fold classes receive the smoothing mass. The score is a leakage-free
cross-validated pseudo-log-likelihood, not an exact Bayesian likelihood.

### Regression

`regression_cv_score` uses `predict()` and a Gaussian residual likelihood. The
variance is estimated from the training fold only:

```text
sigma2 = max(var(y_train), epsilon**2)
log_score = -0.5 * (log(2*pi*sigma2) + residual**2 / sigma2)
```

The mean validation score is used for model weighting; predictions remain the
weighted average of fitted estimator predictions.

## Weights, prediction, and diagnostics

For ordinary prior sampling, posterior ensemble weights are the stable softmax
of CV pseudo-log-likelihoods:

```text
log_importance_weight = cv_log_pseudo_likelihood
posterior_weight      = softmax(log_importance_weight / temperature)
```

The recorded prior is not multiplied into the weight a second time. Sampling
frequency already represents the prior. `temperature < 1` concentrates mass on
better-scoring models; `temperature > 1` spreads mass more evenly.

For classification, predictions are the weighted average of globally aligned
probability vectors. For regression, predictions are the weighted average of
model predictions.

`get_model_draws()` returns serializable dictionaries containing generic fields:

```text
family_name
family_prior_probability
parameters
parameter_prior
representation_family
representation_family_probability
projection_dimension
projection_parameters
subset_size
subset_indices
projection_scale_draw
subset_scale_draw
log_prior
log_proposal
cv_log_pseudo_likelihood
log_importance_weight
posterior_weight
```

Family-specific information belongs inside `parameters` and
`parameter_prior.metadata`; the core does not require flattened fields such as
`n_neighbors` or `covariance_structure`.

`get_model_masses()` returns posterior shares by family and parameter. The
method name is retained for API stability:

```python
{
    "family": {"gaussian": ..., "knn": ..., "linear": ...},
    "parameter": {
        "gaussian": {"covariance_structure": {...}},
        "knn": {"n_neighbors": {...}},
        ...,
    },
}
```

Family posterior shares sum to one. Parameter shares are conditional within each family,
which makes diagnostics meaningful even when families have different numbers
of parameter choices.

## Convergence and parallelism

With `n_estimators="auto"`, the ensemble grows as `20, 40, 80, ...` up to
`max_estimators`. Existing draws are reused. A fixed random subset of the
training data is used to compare successive ensemble predictions. The selected
metric is maximum, mean, or median absolute change. The estimator stores:

- `convergence_history_`;
- `n_estimators_`;
- `converged_`;
- `convergence_subset_indices_`.

Every draw receives a deterministic child seed derived from the base seed and
draw index. This makes serial and parallel fits reproducible and allows
automatic growth to reuse the same earlier draws. Joblib parallelizes model
preparation, scoring, final fitting, and prediction; inner estimators such as
k-NN use `n_jobs=1` to avoid nested parallelism.

## Multiclass visualization

`experiments/classification_2d.py` provides reusable dataset experiments and
`plot_probability_heatmap`.

- Binary problems retain per-class probability heatmaps with dotted contours
  at `dotted_threshold` and `1 - dotted_threshold`.
- Multiclass problems use one panel. Each class has a `tab10` color, and the
  RGB color is the probability-weighted class-color mixture.
- Normalized entropy controls saturation: confident class predictions are
  strongly colored, while uncertain mixtures fade toward white.
- Black contours show argmax class boundaries; the dotted contour marks
  confidence `1 - dotted_threshold`.
- A grayscale confidence colorbar explains the whitening scale.

The notebook is intentionally thin: it runs experiments, prints compact
diagnostics, and displays the returned figures. Plot parameters, including the
dotted threshold, remain notebook-configurable.

## Extending the system

To add a family:

1. implement the adapter protocol;
2. declare supported tasks and representations;
3. implement the complete conditional parameter prior;
4. build a fresh sklearn estimator from the sampled parameters;
5. ensure classification estimators expose `predict_proba`;
6. register the adapter with `FamilyRegistration`.

No changes to `base.py`, `sampling.py`, `scoring.py`, `models.py`, or the
public estimator classes should be required. Add tests for task and
representation validation, deterministic sampling, exact joint prior logging,
class alignment, scoring, serialization, sklearn cloning, and deterministic
parallel fitting.

## Verification

Run the development checks with:

```bash
python -m pytest
ruff check bayesian_model_averaging tests
```

The architecture is considered healthy when a custom adapter can be registered
at runtime and used for classification or regression without modifying the
Bayesian averaging core.
