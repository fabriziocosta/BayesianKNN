# Bayesian Monte Carlo k-NN

`bayesian-knn` provides scikit-learn-compatible Bayesian model averaging over
many weighted k-nearest-neighbour models. Instead of selecting one projection,
subset size, or neighbourhood size, it samples those choices and weights the
resulting models by cross-validated predictive performance.

The full design specification is in [specs.md](specs.md).

## Installation

Install the published package:

```bash
python -m pip install bayesian-knn
```

Install the current checkout for development:

```bash
python -m pip install -e ".[test,dev]"
```

Runtime dependencies are NumPy, SciPy, scikit-learn, and joblib. Python 3.10+
is supported.

## Quick start

### Classification

```python
from bayesian_knn import BayesianKNNClassifier

model = BayesianKNNClassifier(
    n_estimators=40,
    random_state=7,
    n_jobs=-1,
)
model.fit(X_train, y_train)

labels = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

### Regression

```python
from bayesian_knn import BayesianKNNRegressor

model = BayesianKNNRegressor(
    n_estimators=40,
    random_state=7,
    n_jobs=-1,
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
score = model.score(X_test, y_test)
```

Both estimators support `n_estimators="auto"`, which starts with 20 models and
doubles the Monte Carlo ensemble until convergence or `max_estimators` is
reached. The default representation is Gaussian random projection, with
distance-weighted Euclidean k-NN and five-fold cross-validation.

## Useful parameters

- `representation`: `"gaussian"`, `"sparse"`, or `"identity"`.
- `scale_prior`: a `LogisticScalePrior` or a configuration mapping such as
  `{"beta_shape": 2.0, "beta_scale": 1.0}`.
- `min_subset_size`, `max_subset_size`: bounds for sampled training subsets.
- `max_neighbors`: upper bound for sampled neighbourhood sizes.
- `weights` and `metric`: passed to scikit-learn k-NN; defaults are
  `"distance"` and `"euclidean"`.
- `n_estimators`, `max_estimators`, `tolerance`, and `convergence_metric`:
  Monte Carlo ensemble controls.
- `random_state`: makes model draws and parallel fits reproducible.

After fitting, `get_model_draws()` returns dictionaries containing each sampled
representation, subset, neighbourhood, prior draw, pseudo-likelihood, and
posterior weight.

## Development

Run the test suite and lint checks from the repository root:

```bash
python -m pytest
ruff check bayesian_knn tests
```

The GitHub Actions workflow runs both checks across supported Python versions.
