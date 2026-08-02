"""Comparison experiments against common scikit-learn classifiers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..classifier import BayesianKNNClassifier

DATASET_LOADERS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
    "digits": load_digits,
}


@dataclass
class DatasetComparisonResult:
    """Evaluation results for one dataset and four classifiers."""

    dataset: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    models: dict[str, Any]
    scores: dict[str, dict[str, float]]


def load_standard_dataset(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one of the bundled scikit-learn classification datasets."""

    try:
        bunch = DATASET_LOADERS[dataset.lower()]()
    except (AttributeError, KeyError) as error:
        choices = ", ".join(DATASET_LOADERS)
        raise ValueError(f"dataset must be one of: {choices}") from error
    return np.asarray(bunch.data, dtype=float), np.asarray(bunch.target)


def _make_models(
    *,
    bayesian_parameters: dict[str, Any] | None,
    n_neighbors: int,
    random_state: int,
    random_forest_parameters: dict[str, Any] | None,
    svm_parameters: dict[str, Any] | None,
) -> dict[str, Any]:
    bayesian_options = {
        "representation": "gaussian",
        "n_estimators": 20,
        "cv": 5,
        "weights": "distance",
        "n_jobs": 1,
        "random_state": random_state,
    }
    if bayesian_parameters is not None:
        bayesian_options.update(bayesian_parameters)
    forest_options = {
        "n_estimators": 300,
        "random_state": random_state,
        "n_jobs": -1,
    }
    if random_forest_parameters is not None:
        forest_options.update(random_forest_parameters)
    svm_options = {
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
    }
    if svm_parameters is not None:
        svm_options.update(svm_parameters)
    return {
        "Bayesian k-NN": make_pipeline(StandardScaler(), BayesianKNNClassifier(**bayesian_options)),
        "k-NN": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance"),
        ),
        "Random forest": RandomForestClassifier(**forest_options),
        "SVM": make_pipeline(StandardScaler(), SVC(**svm_options)),
    }


def run_dataset_comparison(
    dataset: str = "iris",
    *,
    test_size: float = 0.25,
    random_state: int = 7,
    n_neighbors: int = 15,
    bayesian_parameters: dict[str, Any] | None = None,
    random_forest_parameters: dict[str, Any] | None = None,
    svm_parameters: dict[str, Any] | None = None,
) -> DatasetComparisonResult:
    """Fit and evaluate four classifiers on one standard dataset."""

    X, y = load_standard_dataset(dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    models = _make_models(
        bayesian_parameters=bayesian_parameters,
        n_neighbors=n_neighbors,
        random_state=random_state,
        random_forest_parameters=random_forest_parameters,
        svm_parameters=svm_parameters,
    )
    scores: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        scores[name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
            "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        }
    return DatasetComparisonResult(
        dataset=dataset.lower(),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        models=models,
        scores=scores,
    )


def run_comparison_suite(
    datasets: Iterable[str] = ("iris", "wine", "breast_cancer"),
    **comparison_parameters: Any,
) -> dict[str, DatasetComparisonResult]:
    """Run the same classifier comparison for several datasets."""

    return {
        dataset: run_dataset_comparison(dataset, **comparison_parameters)
        for dataset in datasets
    }


def format_comparison_table(results: dict[str, DatasetComparisonResult]) -> str:
    """Format comparison results as a compact Markdown table."""

    lines = [
        "| Dataset | Classifier | Accuracy | Balanced accuracy | Macro-F1 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for dataset, result in results.items():
        for classifier, metrics in result.scores.items():
            lines.append(
                f"| {dataset} | {classifier} | {metrics['accuracy']:.3f} | "
                f"{metrics['balanced_accuracy']:.3f} | {metrics['macro_f1']:.3f} |"
            )
    return "\n".join(lines)


def plot_comparison_results(
    results: dict[str, DatasetComparisonResult],
    *,
    metric: str = "accuracy",
    output_path: str | None = None,
) -> Any:
    """Plot grouped classifier scores for each dataset."""

    import matplotlib.pyplot as plt

    valid_metrics = {"accuracy", "balanced_accuracy", "macro_f1"}
    if metric not in valid_metrics:
        choices = ", ".join(sorted(valid_metrics))
        raise ValueError(f"metric must be one of: {choices}")
    if not results:
        raise ValueError("results must contain at least one dataset")

    datasets = list(results)
    classifiers = list(next(iter(results.values())).scores)
    positions = np.arange(len(datasets))
    width = 0.8 / len(classifiers)
    fig, ax = plt.subplots(figsize=(max(10, 2.2 * len(datasets)), 6.5))
    for index, classifier in enumerate(classifiers):
        values = [result.scores[classifier][metric] for result in results.values()]
        bars = ax.bar(
            positions + (index - (len(classifiers) - 1) / 2) * width,
            values,
            width,
            label=classifier,
        )
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    ax.set_xticks(positions, datasets)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title("Bayesian k-NN versus standard classifiers")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig
