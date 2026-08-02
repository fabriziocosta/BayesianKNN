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
    """Repeated evaluation results for one dataset and four classifiers.

    ``scores`` contains the mean metric across repeats and ``score_stds``
    contains the corresponding sample standard deviations. ``score_runs``
    retains the individual metric values for further analysis.
    """

    dataset: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    models: dict[str, Any]
    scores: dict[str, dict[str, float]]
    score_stds: dict[str, dict[str, float]]
    score_runs: dict[str, dict[str, tuple[float, ...]]]


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
        "representation": "mixed",
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
    n_repeats: int = 1,
    n_neighbors: int = 15,
    bayesian_parameters: dict[str, Any] | None = None,
    random_forest_parameters: dict[str, Any] | None = None,
    svm_parameters: dict[str, Any] | None = None,
) -> DatasetComparisonResult:
    """Fit and evaluate four classifiers over repeated train/test splits.

    The reported scores are means over ``n_repeats`` stratified holdout
    splits. Standard deviations use ``ddof=1`` and are zero for one repeat.
    """

    if not isinstance(n_repeats, (int, np.integer)) or n_repeats < 1:
        raise ValueError("n_repeats must be a positive integer")

    X, y = load_standard_dataset(dataset)
    seeds = np.random.SeedSequence(random_state).generate_state(
        int(n_repeats), dtype=np.uint32
    )
    models: dict[str, Any] = {}
    X_train = X_test = y_train = y_test = None
    score_values: dict[str, dict[str, list[float]]] = {}
    for repeat_seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=y,
            random_state=int(repeat_seed),
        )
        models = _make_models(
            bayesian_parameters=bayesian_parameters,
            n_neighbors=n_neighbors,
            random_state=int(repeat_seed),
            random_forest_parameters=random_forest_parameters,
            svm_parameters=svm_parameters,
        )
        for name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            metrics = {
                "accuracy": float(accuracy_score(y_test, predictions)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
                "macro_f1": float(f1_score(y_test, predictions, average="macro")),
            }
            if name not in score_values:
                score_values[name] = {metric: [] for metric in metrics}
            for metric, value in metrics.items():
                score_values[name][metric].append(value)

    scores = {
        name: {
            metric: float(np.mean(values))
            for metric, values in metrics.items()
        }
        for name, metrics in score_values.items()
    }
    score_stds = {
        name: {
            metric: float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            for metric, values in metrics.items()
        }
        for name, metrics in score_values.items()
    }
    score_runs = {
        name: {metric: tuple(values) for metric, values in metrics.items()}
        for name, metrics in score_values.items()
    }
    return DatasetComparisonResult(
        dataset=dataset.lower(),
        # These are the final repeat's split and fitted models, retained for
        # interactive inspection while aggregate metrics cover all repeats.
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        models=models,
        scores=scores,
        score_stds=score_stds,
        score_runs=score_runs,
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
        "| Dataset | Classifier | Accuracy (mean ± std) | "
        "Balanced accuracy (mean ± std) | Macro-F1 (mean ± std) |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for dataset, result in results.items():
        for classifier, metrics in result.scores.items():
            stds = result.score_stds[classifier]
            formatted = [
                f"{metrics[metric]:.3f} ± {stds[metric]:.3f}"
                for metric in ("accuracy", "balanced_accuracy", "macro_f1")
            ]
            lines.append(f"| {dataset} | {classifier} | {' | '.join(formatted)} |")
    return "\n".join(lines)


def comparison_results_dataframe(results: dict[str, DatasetComparisonResult]) -> Any:
    """Return repeated comparison results as a pandas DataFrame.

    The DataFrame contains separate mean and standard-deviation columns so
    the values remain numeric and can be sorted or plotted directly.
    """

    import pandas as pd

    if not results:
        raise ValueError("results must contain at least one dataset")

    rows = []
    for dataset, result in results.items():
        for classifier, metrics in result.scores.items():
            standard_deviations = result.score_stds[classifier]
            rows.append(
                {
                    "dataset": dataset,
                    "classifier": classifier,
                    **{
                        f"{metric}_mean": metrics[metric]
                        for metric in ("accuracy", "balanced_accuracy", "macro_f1")
                    },
                    **{
                        f"{metric}_std": standard_deviations[metric]
                        for metric in ("accuracy", "balanced_accuracy", "macro_f1")
                    },
                }
            )
    return (
        pd.DataFrame(rows)
        .set_index(["dataset", "classifier"])
        .sort_index()
        .round(4)
    )


def plot_comparison_results(
    results: dict[str, DatasetComparisonResult],
    *,
    metric: str = "accuracy",
    output_path: str | None = None,
) -> Any:
    """Plot mean grouped classifier scores with repeat standard deviations."""

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
    upper_bound = 1.05
    for index, classifier in enumerate(classifiers):
        values = [result.scores[classifier][metric] for result in results.values()]
        standard_deviations = [
            result.score_stds[classifier][metric] for result in results.values()
        ]
        upper_bound = max(
            upper_bound,
            *(
                value + standard_deviation
                for value, standard_deviation in zip(values, standard_deviations)
            ),
        )
        bars = ax.bar(
            positions + (index - (len(classifiers) - 1) / 2) * width,
            values,
            width,
            yerr=standard_deviations,
            capsize=4,
            error_kw={"elinewidth": 1, "capthick": 1},
            label=classifier,
        )
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    ax.set_xticks(positions, datasets)
    ax.set_ylim(0, max(1.05, 1.08 * upper_bound))
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title("Bayesian k-NN versus standard classifiers")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig
