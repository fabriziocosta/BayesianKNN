"""Reusable 2D classification experiments and probability heat maps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import load_iris, make_blobs, make_moons
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

DATASET_ALIASES = {
    "moon": "moon",
    "moons": "moon",
    "iris": "iris",
    "gaussian": "gaussian",
    "gaussians": "gaussian",
    "two_gaussians": "gaussian",
    "2 equal isotropic gaussians": "gaussian",
    "2_equal_isotropic_gaussians": "gaussian",
    "two_equal_isotropic_gaussians": "gaussian",
    "blob": "blobs",
    "blobs": "blobs",
    "gaussian_blobs": "blobs",
}


@dataclass
class Classification2DResult:
    """Outputs from one 2D classification experiment."""

    dataset: str
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
    probability_class: Any

    @property
    def effective_models(self) -> float:
        return float(1.0 / np.sum(self.model_weights**2))

    @property
    def largest_weight(self) -> float:
        return float(np.max(self.model_weights))


def _canonical_dataset(dataset: str) -> str:
    try:
        return DATASET_ALIASES[dataset.lower()]
    except (AttributeError, KeyError) as error:
        choices = ", ".join(sorted({"moon", "iris", "gaussian", "blobs"}))
        raise ValueError(f"dataset must be one of: {choices}") from error


def _make_equal_isotropic_gaussians(
    n_samples: int,
    mean_distance: float,
    standard_deviation: float,
    random_state: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    if mean_distance <= 0 or standard_deviation <= 0:
        raise ValueError("mean_distance and standard_deviation must be positive")
    rng = np.random.default_rng(random_state)
    counts = (n_samples // 2, n_samples - n_samples // 2)
    covariance = np.eye(2) * standard_deviation**2
    centers = ((-mean_distance / 2, 0.0), (mean_distance / 2, 0.0))
    X = np.vstack(
        [
            rng.multivariate_normal(center, covariance, size=count)
            for center, count in zip(centers, counts)
        ]
    )
    y = np.repeat((0, 1), counts)
    order = rng.permutation(n_samples)
    return X[order], y[order]


def _make_blobs(
    n_samples: int,
    n_classes: int,
    center_radius: float,
    cluster_standard_deviation: float,
    random_state: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(n_classes, (int, np.integer)) or isinstance(n_classes, bool):
        raise ValueError("n_classes must be an integer")
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2")
    if n_samples < n_classes:
        raise ValueError("n_samples must be at least n_classes")
    if center_radius <= 0:
        raise ValueError("center_radius must be positive")
    if cluster_standard_deviation <= 0:
        raise ValueError("cluster_standard_deviation must be positive")
    angles = np.linspace(0.0, 2.0 * np.pi, int(n_classes), endpoint=False)
    centers = center_radius * np.column_stack((np.cos(angles), np.sin(angles)))
    return make_blobs(
        n_samples=n_samples,
        centers=centers,
        n_features=2,
        cluster_std=cluster_standard_deviation,
        random_state=random_state,
    )


def make_2d_dataset(
    dataset: str = "moon",
    *,
    n_samples: int = 600,
    noise: float = 0.24,
    random_state: int | None = 7,
    feature_indices: tuple[int, int] = (0, 1),
    mean_distance: float = 3.0,
    standard_deviation: float = 1.0,
    n_classes: int = 3,
    blob_center_radius: float = 3.0,
    blob_cluster_standard_deviation: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Create one of the supported two-dimensional classification datasets.

    ``n_classes`` controls the number of classes for the ``"blobs"`` dataset.
    Blob centers are placed evenly on a circle so their separation is
    controlled by ``blob_center_radius`` rather than by a random center box.
    """

    name = _canonical_dataset(dataset)
    if name == "moon":
        return make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    if name == "iris":
        iris = load_iris()
        if len(feature_indices) != 2 or len(set(feature_indices)) != 2:
            raise ValueError("feature_indices must contain two distinct feature indices")
        if min(feature_indices) < 0 or max(feature_indices) >= iris.data.shape[1]:
            raise ValueError("Iris feature_indices must be between 0 and 3")
        return iris.data[:, feature_indices], iris.target
    if name == "blobs":
        return _make_blobs(
            n_samples=n_samples,
            n_classes=n_classes,
            center_radius=blob_center_radius,
            cluster_standard_deviation=blob_cluster_standard_deviation,
            random_state=random_state,
        )
    return _make_equal_isotropic_gaussians(
        n_samples=n_samples,
        mean_distance=mean_distance,
        standard_deviation=standard_deviation,
        random_state=random_state,
    )


def run_2d_classification_experiment(
    dataset: str = "moon",
    *,
    dataset_parameters: dict[str, Any] | None = None,
    test_size: float = 0.30,
    split_random_state: int = 7,
    model_parameters: dict[str, Any] | None = None,
) -> Classification2DResult:
    """Generate, fit, and evaluate one supported 2D classification dataset."""

    parameters = {
        "n_samples": 600,
        "noise": 0.24,
        "random_state": 7,
        "feature_indices": (0, 1),
        "mean_distance": 3.0,
        "standard_deviation": 1.0,
        "n_classes": 3,
        "blob_center_radius": 3.0,
        "blob_cluster_standard_deviation": 1.5,
    }
    if dataset_parameters is not None:
        parameters.update(dataset_parameters)
    name = _canonical_dataset(dataset)
    X, y = make_2d_dataset(name, **parameters)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=split_random_state,
    )
    model_options = dict(DEFAULT_MODEL_PARAMETERS)
    if model_parameters is not None:
        model_options.update(model_parameters)
    model = BayesianKNNClassifier(**model_options)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model_weights = np.asarray(
        [draw["posterior_weight"] for draw in model.get_model_draws()], dtype=float
    )
    probability_class = model.classes_[1]
    return Classification2DResult(
        dataset=name,
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
        probability_class=probability_class,
    )


def format_convergence_history(result: Classification2DResult) -> str:
    """Format the ensemble growth and probability-change history."""

    lines = ["20 estimators: initial ensemble"]
    for entry in result.model.convergence_history_:
        lines.append(
            f"{int(entry['n_estimators'])} estimators: "
            f"median probability change = {entry['median_absolute_change']:.6f}"
        )
    return "\n".join(lines)


def _class_probability_index(result: Classification2DResult, class_label: Any | None) -> int:
    label = result.probability_class if class_label is None else class_label
    matches = np.flatnonzero(result.model.classes_ == label)
    if len(matches) == 0:
        raise ValueError(f"class_label {label!r} is not present in the fitted model")
    return int(matches[0])


def probability_surface(
    result: Classification2DResult,
    *,
    padding: float = 0.55,
    grid_size: int = 350,
    class_label: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate one class probability over a rectangular feature-space grid."""

    xx, yy, probabilities = _probability_grid(
        result,
        padding=padding,
        grid_size=grid_size,
    )
    class_index = _class_probability_index(result, class_label)
    return xx, yy, probabilities[:, :, class_index]


def _probability_grid(
    result: Classification2DResult,
    *,
    padding: float,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_min, x_max = result.X[:, 0].min() - padding, result.X[:, 0].max() + padding
    y_min, y_max = result.X[:, 1].min() - padding, result.X[:, 1].max() + padding
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_size),
        np.linspace(y_min, y_max, grid_size),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    probabilities = result.model.predict_proba(grid).reshape(
        xx.shape + (len(result.model.classes_),)
    )
    return xx, yy, probabilities


def plot_probability_heatmap(
    result: Classification2DResult,
    *,
    padding: float = 0.55,
    grid_size: int = 350,
    class_label: Any | None = None,
    output_path: str | Path | None = None,
) -> Any:
    """Plot probability surfaces, one panel per class for multiclass data."""

    import matplotlib.pyplot as plt

    xx, yy, probabilities = _probability_grid(
        result,
        padding=padding,
        grid_size=grid_size,
    )
    classes = result.model.classes_
    labels = list(classes) if class_label is None and len(classes) > 2 else [
        result.probability_class if class_label is None else class_label
    ]
    indices = [_class_probability_index(result, label) for label in labels]
    figure_width = 9 if len(labels) == 1 else 6 * len(labels)
    fig, axes = plt.subplots(1, len(labels), figsize=(figure_width, 6.5), squeeze=False)
    axes = axes.ravel()
    colors = plt.get_cmap("tab10")
    predicted_indices = np.argmax(probabilities, axis=2)
    for ax, label, index in zip(axes, labels, indices):
        probability = probabilities[:, :, index]
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
        if len(classes) == 2:
            ax.contour(xx, yy, probability, levels=[0.5], linewidths=2, colors="black")
        else:
            boundaries = np.arange(0.5, len(classes) - 0.5, 1.0)
            ax.contour(xx, yy, predicted_indices, levels=boundaries, linewidths=2, colors="black")
        for class_index, class_value in enumerate(classes):
            mask = result.y_train == class_value
            ax.scatter(
                result.X_train[mask, 0],
                result.X_train[mask, 1],
                s=24,
                color=colors(class_index),
                edgecolors="white",
                linewidths=0.45,
                label=f"Class {class_value}",
            )
        colorbar = fig.colorbar(heat, ax=ax)
        colorbar.set_label(f"Predicted probability of class {label}")
        ax.set_title(f"Class {label} probability")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.legend(loc="upper right")
    fig.suptitle(
        f"Bayesian Monte Carlo k-NN on {result.dataset}\n"
        f"{result.model.n_estimators_} estimators, test accuracy = {result.test_accuracy:.3f}"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    if output_path is not None:
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig
