"""Reproducible two-moons experiment and probability heat map."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from ..classifier import BayesianKNNClassifier

DEFAULT_MODEL_PARAMETERS: dict[str, Any] = {
    "n_estimators": "auto",
    "max_estimators": 640,
    "tolerance": 0.01,
    "convergence_metric": "median",
    "convergence_size": 256,
    "cv": 5,
    "max_neighbors": None,
    "weights": "distance",
    "n_jobs": -1,
    "random_state": 12,
}


@dataclass
class TwoMoonsResult:
    """Outputs from one complete two-moons run."""

    X: np.ndarray
    y: np.ndarray
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    model: BayesianKNNClassifier
    y_pred: np.ndarray
    test_accuracy: float
    model_weights: np.ndarray

    @property
    def effective_models(self) -> float:
        return float(1.0 / np.sum(self.model_weights**2))

    @property
    def largest_weight(self) -> float:
        return float(np.max(self.model_weights))


def run_two_moons_experiment(
    *,
    n_samples: int = 600,
    noise: float = 0.24,
    data_random_state: int = 7,
    test_size: float = 0.30,
    split_random_state: int = 7,
    model_parameters: dict[str, Any] | None = None,
) -> TwoMoonsResult:
    """Generate, fit, and evaluate the reproducible two-moons experiment."""

    X, y = make_moons(
        n_samples=n_samples,
        noise=noise,
        random_state=data_random_state,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=split_random_state,
    )
    parameters = dict(DEFAULT_MODEL_PARAMETERS)
    if model_parameters is not None:
        parameters.update(model_parameters)
    model = BayesianKNNClassifier(**parameters)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model_weights = np.asarray(
        [draw["posterior_weight"] for draw in model.get_model_draws()], dtype=float
    )
    return TwoMoonsResult(
        X=X,
        y=y,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        model=model,
        y_pred=y_pred,
        test_accuracy=float(accuracy_score(y_test, y_pred)),
        model_weights=model_weights,
    )


def format_convergence_history(result: TwoMoonsResult) -> str:
    """Format the ensemble growth and median-change history for display."""

    lines = ["20 estimators: initial ensemble"]
    for entry in result.model.convergence_history_:
        lines.append(
            f"{int(entry['n_estimators'])} estimators: "
            f"median probability change = {entry['median_absolute_change']:.6f}"
        )
    return "\n".join(lines)


def probability_surface(
    result: TwoMoonsResult,
    *,
    padding: float = 0.55,
    grid_size: int = 350,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate class-1 probability over a rectangular feature-space grid."""

    x_min, x_max = result.X[:, 0].min() - padding, result.X[:, 0].max() + padding
    y_min, y_max = result.X[:, 1].min() - padding, result.X[:, 1].max() + padding
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_size),
        np.linspace(y_min, y_max, grid_size),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    probability = result.model.predict_proba(grid)[:, 1].reshape(xx.shape)
    return xx, yy, probability


def plot_probability_heatmap(
    result: TwoMoonsResult,
    *,
    padding: float = 0.55,
    grid_size: int = 350,
    output_path: str | Path | None = "moons_bayesian_knn_probability_heatmap.png",
) -> Any:
    """Plot and optionally save the model-averaged two-moons probability surface."""

    import matplotlib.pyplot as plt

    xx, yy, probability = probability_surface(result, padding=padding, grid_size=grid_size)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    heat = ax.contourf(
        xx,
        yy,
        probability,
        levels=np.linspace(0, 1, 41),
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        alpha=0.90,
    )
    ax.contour(xx, yy, probability, levels=[0.5], linewidths=2, colors="black")
    ax.scatter(
        result.X_train[result.y_train == 0, 0],
        result.X_train[result.y_train == 0, 1],
        s=24,
        c="#2166ac",
        edgecolors="white",
        linewidths=0.45,
        label="Class 0",
    )
    ax.scatter(
        result.X_train[result.y_train == 1, 0],
        result.X_train[result.y_train == 1, 1],
        s=24,
        c="#b2182b",
        edgecolors="white",
        linewidths=0.45,
        label="Class 1",
    )
    colorbar = fig.colorbar(heat, ax=ax)
    colorbar.set_label("Predicted probability of class 1\nblue = class 0, red = class 1")
    ax.set_title(
        "Bayesian Monte Carlo k-NN on the two-moons dataset\n"
        f"{result.model.n_estimators_} estimators, 5-fold CV, "
        f"test accuracy = {result.test_accuracy:.3f}"
    )
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend(loc="upper right")
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig
