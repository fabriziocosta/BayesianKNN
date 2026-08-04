# Bayesian Predictive Model Averaging (BPMA) Architecture

This document describes the implemented BPMA architecture. BPMA is a
Bayesian-inspired predictive model averaging framework that samples complete
predictive models from declared priors, estimates predictive evidence with
cross-validation, and averages their predictions. It does not optimize one
hyperparameter configuration.

Classical Bayesian Model Averaging uses marginal likelihoods to obtain
posterior model probabilities. BPMA instead uses cross-validated predictive
evidence, which makes the framework applicable to arbitrary machine-learning
estimators whose exact marginal likelihoods are inaccessible or intractable.
The methodological overview is in [WHITE_PAPER.md](WHITE_PAPER.md).

## Design

The core integrates three kinds of uncertainty:

1. estimator family;
2. training subset;
3. estimator-family parameters.

Each Monte Carlo draw is a complete, fitted model. The draw records its prior
and cross-validated score, while the ensemble weight is determined by the
cross-validated predictive evidence. This keeps the BPMA integration engine generic:
family-specific parameter logic lives in adapters, and scoring only depends on
the estimator contract.

The fit pipeline is:

```text
validate data and CV
        |
normalize runtime family registry
        |
sample family
        |
sample CV-admissible subset -> construct SamplingContext
        |
adapter samples complete parameters
        |
generic CV score -> fit final estimator -> store ModelDraw
        |
stable softmax over CV scores -> predictions, predictive shares, convergence
```

## Package layout

```text
bayesian_predictive_model_averaging/
  __init__.py          public exports
  base.py              shared sklearn estimator and ensemble engine
  classifier.py        BayesianPredictiveModelAveragingClassifier
  regressor.py         BayesianPredictiveModelAveragingRegressor
  adapters.py          adapter protocol, registrations, built-in adapters
  priors.py            reusable prior objects and draw records
  sampling.py          complete-model sampling, CV subsets, fitting
  scoring.py           generic classification and regression scoring
  models.py            ModelDraw and predictive-share diagnostics
  model_families.py    GaussianClassifier implementation
  convergence.py       prediction-change convergence metrics
  utils.py             deterministic seeds and numerical helpers
  experiments/
    classification_2d.py
```

## Estimator API

The public estimators are:

```python
from bayesian_predictive_model_averaging import (
    BayesianPredictiveModelAveragingClassifier,
    BayesianPredictiveModelAveragingRegressor,
)

classifier = BayesianPredictiveModelAveragingClassifier(
    n_estimators=40,
    random_state=7,
)
classifier.fit(X_train, y_train)
probabilities = classifier.predict_proba(X_test)
predictions = classifier.predict(X_test)

regressor = BayesianPredictiveModelAveragingRegressor(n_estimators=40, random_state=7)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)
```

Both estimators expose `fit`, `predict`, `score`, and `get_model_draws`.
Classifiers additionally expose `predict_proba`; regressors intentionally do
not. The estimators preserve constructor parameters through sklearn cloning,
pipelines, and `GridSearchCV`.

Important constructor parameters are:

- `family_registry`: runtime estimator-family registrations;
- `scale_prior`: the prior used for subset size and ordered adapter parameters;
- `min_subset_size`, `max_subset_size`, and `cv`;
- `n_estimators`, or `"auto"` with `max_estimators`;
- `tolerance`, `convergence_metric`, and `convergence_size`;
- `alpha` for classification probability smoothing;
- `epsilon` for the regression variance floor;
- `temperature` for target pseudo-posterior concentration;
- `n_jobs` and `random_state`.

Adaptive importance sampling is disabled by default. When enabled, the main
controls are `round_size`, `min_rounds`, `max_rounds`,
`defensive_prior_weight`, `proposal_tolerance`, optional
`prediction_tolerance`, optional `ess_target_fraction`, `stopping_patience`,
and `adaptation_temperature`. In adaptive mode `max_estimators` is the total
draw budget; `n_estimators` retains its existing meaning when adaptation is
disabled.

## Runtime family adapters

Family selection is registry-driven. The core does not branch on family names.
The public interfaces are:

```python
class EstimatorFamilyAdapter(Protocol):
    name: str
    supported_tasks: frozenset[str]

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
specified, the default is a uniform mixture of k-NN, gated linear-mixture,
Gaussian-mixture, MLP, and decision-tree families.

### Adapter responsibilities

An adapter owns all family-specific parameter logic. `sample_parameters` gets
a `SamplingContext` containing task, input feature count, class count,
subset size, minimum CV training-fold size, classes, and the shared scale
prior. For classification it also contains the minimum per-class CV training
size and minimum distinct per-class training size, which lets class-conditional
adapters enforce valid configurations. It must return one `ParameterDraw`
containing:

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

| Adapter | Classification | Regression | Main parameters |
| --- | --- | --- | --- |
| `KNNAdapter` | `KNeighborsClassifier` | `KNeighborsRegressor` | `n_neighbors`, `weights`, `metric` |
| `LinearAdapter` | `LogisticRegression` | `Ridge` | solver/iterations or ridge `alpha` |
| `GaussianAdapter` | `GaussianClassifier` | `BayesianRidge` | classification covariance structure |
| `GaussianMixtureAdapter` | class-conditional `GaussianMixture` | — | `n_components` up to 30, covariance structure |
| `LinearMixtureAdapter` | gated logistic experts | gated ridge experts | expert count, expert/gate regularization |
| `MLPAdapter` | `MLPClassifier` | `MLPRegressor` | architecture, activation, regularization, learning rate |
| `DecisionTreeAdapter` | `DecisionTreeClassifier` | `DecisionTreeRegressor` | depth, split/leaf sizes, criterion, splitter |

The `GaussianAdapter` fits a single Gaussian likelihood for every class. The
default `GaussianMixtureAdapter` instead fits a separate Gaussian mixture for
every class, with a simplicity prior favoring fewer than 30 components. Both
families sample covariance structure from `isotropic`, `diagonal`, or `full`.
Gaussian regression uses `BayesianRidge`; covariance structure is not sampled
for regression. The default registry uses `GaussianMixtureAdapter`; the
single-Gaussian family is available through explicit registration.

All built-in adapters currently return predictive concentration `1.0`. In
particular, k-NN neighbourhood size is already expressed in the k-NN
probability vector and is not applied a second time as a confidence multiplier.
This calibration choice keeps family CV scores comparable.

`LinearMixtureAdapter` is the default linear family. It fits a finite mixture
of logistic or ridge experts with a learned linear softmax gate. The gate makes
expert weights depend on the input, so the resulting predictor can represent
piecewise-linear nonlinear boundaries. Its expert-count prior favors fewer
experts. `LinearAdapter` remains available as the single-expert alternative.

### Registering a new family

A new sklearn family requires one adapter and its priors; the core, scorer,
model record, and estimator classes do not need modification.

```python
class ToyAdapter:
    name = "toy"
    supported_tasks = frozenset({"classification"})
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

model = BayesianPredictiveModelAveragingClassifier(
    family_registry=[FamilyRegistration(ToyAdapter(), prior_weight=1.0)]
)
```

## Priors

`priors.py` contains reusable prior objects. Each prior returns both a sampled
value and the log probability of that value, together with metadata sufficient
for diagnostics.

### Ordered scale prior

`LogisticScalePrior` is shared by subset size and is available to adapters for
ordered parameters such as k-NN neighbourhood size.
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

### Log-scale parameter sweeps

`LogisticLogScalePrior` provides a reusable finite sweep for positive
parameters. It constructs an inclusive geometric grid and applies the same
latent sigmoid preference for lower values as `LogisticScalePrior`, but on the
grid's normalized log positions. For example, the following samples `C` from
`[1e-2, 1e-1, 1, 1e1, 1e2]`, with lower values favored while every scale remains
available:

```python
from bayesian_predictive_model_averaging import (
    LogisticLogScalePrior,
    RecursivePartitionRBFAdapter,
)

adapter = RecursivePartitionRBFAdapter(
    c_prior=LogisticLogScalePrior(low=1e-2, high=1e2, n_values=5)
)
```

The prior records the complete grid and conditional probability vector in the
parameter-draw metadata. Existing `c_values` configurations remain supported;
when `c_prior` is supplied for a recursive-partition SVM adapter, it controls
the `C` sweep and takes precedence over `c_values`.

`MLPAdapter` uses a finite, simplicity-weighted architecture prior rather than
independently sampling incompatible layer fields. Its joint parameter prior
covers `hidden_layer_sizes`, activation (including `logistic`), `alpha`, and
`learning_rate_init`.

## Sampling a complete model

With ordinary sampling, `sampling.py` performs the following sequence for every
deterministic child seed:

1. select a registered family using normalized family weights;
2. validate the selected task;
3. sample an admissible subset size and a subset uniformly conditional on that
   size;
4. construct the CV splitter and `SamplingContext`;
5. ask the adapter for a complete parameter draw;
6. score the configuration through the generic scorer;
7. build and fit the final estimator on the sampled subset;
8. store the resulting `ModelDraw`.

Classification subsets must contain at least `cv` observations from every
global class. Regression subsets must contain at least `cv` observations.
The sampled k-NN neighbourhood size is bounded by the smallest CV training
fold, and invalid configurations fail rather than being silently scored.

The stored prior contribution is:

```text
log_prior =
    log(family prior)
  + subset-size log probability
  + conditional subset log probability
  + adapter-parameter log probability
```

`log_proposal` is equal to `log_prior` because the current sampler proposes
directly from the declared prior.

## Adaptive importance sampling

Adaptive sampling preserves the declared prior as the target distribution. The
first round samples from the prior. Later rounds adapt only estimator-family
probabilities; subset sizes and adapter parameters continue to use their
declared conditional priors. For family `f`:

```text
q_t(f) = epsilon * p(f) + (1 - epsilon) * qhat_t(f)
```

where `qhat_t(f)` is the weighted family mass from earlier draws and `epsilon`
is `defensive_prior_weight`. This defensive mixture keeps every prior-supported
family reachable. `adaptation_temperature` broadens or concentrates `qhat`;
it does not change the target temperature.

All rounds are pooled with a deterministic-mixture proposal. If `alpha_t` is
the fraction of draws generated in round `t`, then:

```text
q_mix(theta) = sum_t alpha_t * q_t(theta)
log_target(theta) = log_prior(theta) + score(theta) / temperature
log_importance_weight(theta) = log_target(theta) - log(q_mix(theta))
```

When adaptation is disabled, the proposal equals the prior and weights reduce
to the existing score-only softmax. The learned proposal is a computational
allocation mechanism, not an updated prior, and improves Monte Carlo
efficiency rather than reducing uncertainty from finite data.

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

For ordinary prior sampling, normalized predictive weights are the stable softmax
of CV pseudo-log-likelihoods:

```text
log_importance_weight = cv_log_pseudo_likelihood
posterior_weight      = softmax(log_importance_weight / temperature)
```

The recorded prior is not multiplied into the weight a second time. Sampling
frequency already represents the prior. `temperature < 1` concentrates mass on
better-scoring models; `temperature > 1` spreads mass more evenly.

Adaptive draws instead use the prior-corrected target/proposal ratio described
above. Every draw records its generating round, proposal identifier, family
proposal probability, generating proposal log density, final mixture proposal
log density, unnormalized importance weight, and normalized predictive weight.

For classification, predictions are the weighted average of globally aligned
probability vectors. For regression, predictions are the weighted average of
model predictions.

`get_model_draws()` returns serializable dictionaries containing generic fields:

```text
family_name
family_prior_probability
round_index
proposal_id
family_proposal_probability
parameters
parameter_prior
subset_size
subset_indices
subset_scale_draw
log_prior
log_proposal
generating_log_proposal
cv_log_pseudo_likelihood
log_importance_weight
posterior_weight
```

Family-specific information belongs inside `parameters` and
`parameter_prior.metadata`; the core does not require flattened fields such as
`n_neighbors` or `covariance_structure`.

`get_model_masses()` returns normalized predictive shares by family and
parameter. The method name is retained for API stability:

```python
{
    "family": {"gaussian_mixture": ..., "knn": ..., "linear_mixture": ..., "mlp": ..., "decision_tree": ...},
    "parameter": {
        "gaussian_mixture": {
            "covariance_structure": {...},
            "n_components": {...},
        },
        "knn": {"n_neighbors": {...}},
        ...,
    },
}
```

Family predictive shares sum to one. Parameter shares are conditional within each family,
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

Adaptive fits additionally expose `proposal_history_`, `round_history_`,
`n_rounds_`, `effective_sample_size_`,
`effective_sample_size_fraction_`, `adaptive_converged_`, and
`stopping_reason_`. Round history records proposal distance, prediction change,
ESS, maximum normalized weight, family proposal probabilities, predictive family
mass, and the active stopping state. Proposal and prediction stability require
the configured number of consecutive successful rounds; ESS is diagnostic unless
an `ess_target_fraction` is configured. Adapter-owned parameter adaptation is
reserved for a future extension; the initial implementation adapts families
only.

Every draw receives a deterministic child seed derived from the base seed and
draw index. This makes serial and parallel fits reproducible and allows
automatic growth to reuse the same earlier draws. Joblib parallelizes model
preparation, scoring, final fitting, and prediction; inner estimators such as
k-NN use `n_jobs=1` to avoid nested parallelism.

## Multiclass visualization

`experiments/classification_2d.py` provides reusable dataset experiments and
`plot_probability_heatmap`.

- Binary problems retain per-class probability heatmaps with thin contours at
  probability `0.5`.
- Multiclass problems use one panel. Each class has a `tab10` color, and the
  RGB color is the probability-weighted class-color mixture.
- Normalized entropy controls saturation: confident class predictions are
  strongly colored, while uncertain mixtures fade toward white.
- Black contours show argmax class boundaries; a thin contour marks confidence
  `0.5`.
- A grayscale confidence colorbar explains the whitening scale.

The notebook is intentionally thin: it runs experiments, prints compact
diagnostics, and displays the returned figures. Padding, grid resolution, and
output path remain notebook-configurable.

For datasets with a known generating rule, the experiment also reports a
Bayes-error reference. The equal-isotropic two-Gaussian dataset uses its
closed-form error; the blobs, anisotropic-blobs, and XOR generators use a
density-based estimate evaluated on the generated sample; and the moon,
circles, spirals, and checkerboard generators use their known noiseless
boundary or curves evaluated on the actual noisy sample. Iris and generic
sklearn classification data report the reference error as unavailable. The
report labels each value as exact, density estimate, or generator oracle, and
includes the model-error/Bayes-error ratio when available.

## Extending the system

To add a family:

1. implement the adapter protocol;
2. declare supported tasks;
3. implement the complete conditional parameter prior;
4. build a fresh sklearn estimator from the sampled parameters;
5. ensure classification estimators expose `predict_proba`;
6. register the adapter with `FamilyRegistration`.

No changes to `base.py`, `sampling.py`, `scoring.py`, `models.py`, or the
public estimator classes should be required. Add tests for task validation,
deterministic sampling, exact joint prior logging,
class alignment, scoring, serialization, sklearn cloning, and deterministic
parallel fitting.

## Verification

Run the development checks with:

```bash
python -m pytest
ruff check bayesian_predictive_model_averaging tests
```

The architecture is considered healthy when a custom adapter can be registered
at runtime and used for classification or regression without modifying the
Bayesian averaging core.
