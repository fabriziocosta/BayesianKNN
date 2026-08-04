# Bayesian Predictive Model Averaging (BPMA)

Scikit-learn-compatible Bayesian Predictive Model Averaging for classification
and regression, with pluggable estimator-family adapters.

## Install

```bash
python -m pip install bayesian-predictive-model-averaging
```

## Quick start

```python
from bayesian_predictive_model_averaging import BayesianPredictiveModelAveragingClassifier

model = BayesianPredictiveModelAveragingClassifier(n_estimators=40, random_state=7)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

The package also provides `BayesianPredictiveModelAveragingRegressor` and
supports custom estimator families, explicit priors, diagnostics, and optional
adaptive importance sampling.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the implementation and extension
details, [WHITE_PAPER.md](WHITE_PAPER.md) for the methodological overview, and
[`notebooks/simple_library_usage.ipynb`](notebooks/simple_library_usage.ipynb)
for a complete example.

## Development

```bash
python -m pip install -e ".[test,dev]"
python -m pytest
ruff check bayesian_predictive_model_averaging tests
```
