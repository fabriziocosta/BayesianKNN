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

The default runtime registry averages k-NN, linear, and Gaussian adapters with
uniform family prior weights. MLP and new scikit-learn estimator families can be
registered explicitly with a prior weight and an adapter-owned hyperparameter
prior without changing the averaging engine.

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

For development:

```bash
python -m pip install -e ".[test,dev]"
python -m pytest
ruff check bayesian_model_averaging tests
```
