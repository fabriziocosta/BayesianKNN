# Bayesian Monte Carlo k-NN

Scikit-learn-compatible Bayesian model averaging for weighted k-nearest
neighbours classifiers and regressors.

```bash
python -m pip install bayesian-knn
```

```python
from bayesian_knn import BayesianKNNClassifier

model = BayesianKNNClassifier(n_estimators=40, random_state=7)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

Available estimators:

- `BayesianKNNClassifier`, including `predict_proba()`
- `BayesianKNNRegressor`

The full design and implementation specification is in [specs.md](specs.md).

For development:

```bash
python -m pip install -e ".[test,dev]"
python -m pytest
ruff check bayesian_knn tests
```
