# Bayesian Model Averaging

Scikit-learn-compatible Bayesian model averaging for k-nearest-neighbours
classifiers and regressors.

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

The default `model_family="mixed"` averages k-NN, linear, and Gaussian models.
Gaussian classifiers also integrate isotropic, diagonal, and full covariance
structures with a simplicity prior.

The full design and implementation documentation is in [ARCHITECTURE.md](ARCHITECTURE.md).

For development:

```bash
python -m pip install -e ".[test,dev]"
python -m pytest
ruff check bayesian_model_averaging tests
```
