# Bayesian Predictive Model Averaging (BPMA)

Scikit-learn-compatible Bayesian Predictive Model Averaging with pluggable
estimator-family adapters for classifiers and regressors.

**Bayesian Predictive Model Averaging (BPMA)** is a Bayesian-inspired ensemble
framework in which models are sampled from prior distributions over model
families and hyperparameters, and weighted according to estimates of their
predictive evidence obtained through cross-validation, rather than exact
marginal likelihoods.

```bash
python -m pip install bayesian-predictive-model-averaging
```

```python
from bayesian_model_averaging import BayesianPredictiveModelAveragingClassifier

model = BayesianPredictiveModelAveragingClassifier(n_estimators=40, random_state=7)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

Available estimators:

- `BayesianPredictiveModelAveragingClassifier`, including `predict_proba()`
- `BayesianPredictiveModelAveragingRegressor`

The default BPMA registry averages k-NN, gated linear-mixture,
Gaussian-mixture, MLP, and decision-tree adapters with uniform family prior
weights. New scikit-learn estimator families can be registered explicitly with
a prior weight and an adapter-owned hyperparameter prior without changing the
averaging engine.

The decision-tree adapter uses priors over tree depth, split and leaf sizes,
criterion, and splitter.

The default Gaussian-mixture adapter fits one mixture per class, samples up to
30 components, prefers simpler mixtures, and samples isotropic, diagonal, or
full covariance structures.

`LinearMixtureAdapter` is the default linear family: a gated mixture of
logistic or ridge linear experts. Its learned linear softmax gate makes the
expert weights input-dependent while retaining linear experts. The simpler
`LinearAdapter` and single-Gaussian `GaussianAdapter` remain available for
explicit registration.

The optional `recursive-partition` extra exposes four non-bagged recursive
partition adapters: `RecursivePartitionLinearAdapter`,
`RecursivePartitionQuadraticAdapter`, `RecursivePartitionRBFAdapter`, and
`RecursivePartitionQDAAdapter`. The SVM adapters use a simplicity-weighted
discrete prior over `C` (smaller `C` is simpler). Install the sister package
with `python -m pip install -e ../RecursiveParitionClassifier` when working
from the sibling repositories, or install the optional extra when the package
is available from your package index.

For example, use only a configured k-NN family with:

```python
from bayesian_model_averaging import BayesianPredictiveModelAveragingClassifier, FamilyRegistration, KNNAdapter

model = BayesianPredictiveModelAveragingClassifier(
    family_registry=[FamilyRegistration(KNNAdapter(max_neighbors=32), prior_weight=1.0)],
    n_estimators=40,
    random_state=7,
)
```

Adaptive importance sampling is opt-in. The declared family weights remain the
target prior; later rounds only change computational allocation through a
defensive proposal, with deterministic-mixture importance correction:

```python
model = BayesianPredictiveModelAveragingClassifier(
    adaptive_importance_sampling=True,
    round_size=50,
    max_estimators=500,
    defensive_prior_weight=0.2,
    proposal_tolerance=1e-3,
    stopping_patience=2,
    random_state=7,
)
```

`temperature` is the target pseudo-posterior temperature. The separate
`adaptation_temperature` controls how concentrated family proposals become.
Adaptive rounds stop when enabled proposal, prediction, and ESS criteria remain
stable for the configured patience, or when the estimator/round budget is hit.
The learned proposal is never treated as an updated prior.

The implementation design is in [ARCHITECTURE.md](ARCHITECTURE.md), and the
methodological overview is in [WHITE_PAPER.md](WHITE_PAPER.md).

For a minimal end-to-end example, see
[`notebooks/simple_library_usage.ipynb`](notebooks/simple_library_usage.ipynb).

For development:

```bash
python -m pip install -e ".[test,dev]"
python -m pytest
ruff check bayesian_model_averaging tests
```
