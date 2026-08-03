# Bayesian Model Averaging

Scikit-learn-compatible Bayesian model averaging with pluggable estimator-family
adapters for classifiers and regressors.

```bash
python -m pip install bayesian-model-averaging
```

```python
from bayesian_model_averaging import BayesianModelAveragingClassifier

model = BayesianModelAveragingClassifier(n_estimators=40, random_state=7)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

Available estimators:

- `BayesianModelAveragingClassifier`, including `predict_proba()`
- `BayesianModelAveragingRegressor`

The default runtime registry averages k-NN, gated linear-mixture,
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
from bayesian_model_averaging import BayesianModelAveragingClassifier, FamilyRegistration, KNNAdapter

model = BayesianModelAveragingClassifier(
    family_registry=[FamilyRegistration(KNNAdapter(max_neighbors=32), prior_weight=1.0)],
    n_estimators=40,
    random_state=7,
)
```

The full design and implementation documentation is in [ARCHITECTURE.md](ARCHITECTURE.md).

For a minimal end-to-end example, see
[`notebooks/simple_library_usage.ipynb`](notebooks/simple_library_usage.ipynb).

For development:

```bash
python -m pip install -e ".[test,dev]"
python -m pytest
ruff check bayesian_model_averaging tests
```
