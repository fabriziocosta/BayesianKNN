"""Reusable 2D classification experiments and probability heat maps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import (
    load_iris,
    make_blobs,
    make_circles,
    make_moons,
)
from sklearn.datasets import (
    make_classification as sklearn_make_classification,
)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from ..adapters import FamilyRegistration, KNNAdapter
from ..classifier import BayesianModelAveragingClassifier

DEFAULT_MODEL_PARAMETERS: dict[str, Any] = {
    "n_estimators": "auto",
    "max_estimators": 640,
    "tolerance": 0.01,
    "convergence_metric": "median",
    "convergence_size": 256,
    "cv": 5,
    "family_registry": [FamilyRegistration(KNNAdapter(), 1.0)],
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
    "circle": "circles",
    "circles": "circles",
    "xor": "xor",
    "spiral": "spirals",
    "spirals": "spirals",
    "anisotropic": "anisotropic_blobs",
    "anisotropic_blobs": "anisotropic_blobs",
    "checker": "checkerboard",
    "checkerboard": "checkerboard",
    "classification": "classification",
    "make_classification": "classification",
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
    model: BayesianModelAveragingClassifier
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

    @property
    def model_masses(self) -> dict[str, Any]:
        """Return posterior mass by model family and family-specific choice."""

        return self.model.get_model_masses()


def _canonical_dataset(dataset: str) -> str:
    try:
        return DATASET_ALIASES[dataset.lower()]
    except (AttributeError, KeyError) as error:
        choices = ", ".join(
            sorted(
                {
                    "moon",
                    "iris",
                    "gaussian",
                    "blobs",
                    "circles",
                    "xor",
                    "spirals",
                    "anisotropic_blobs",
                    "checkerboard",
                    "classification",
                }
            )
        )
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


def _make_xor(
    n_samples: int,
    cluster_standard_deviation: float,
    random_state: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if cluster_standard_deviation <= 0:
        raise ValueError("cluster_standard_deviation must be positive")
    centers = np.array([(-2.0, -2.0), (-2.0, 2.0), (2.0, -2.0), (2.0, 2.0)])
    X, cluster_labels = make_blobs(
        n_samples=n_samples,
        centers=centers,
        cluster_std=cluster_standard_deviation,
        random_state=random_state,
    )
    return X, np.asarray([0, 1, 1, 0])[cluster_labels]


def _make_spirals(
    n_samples: int,
    turns: float,
    noise: float,
    random_state: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if turns <= 0:
        raise ValueError("spiral_turns must be positive")
    if noise < 0:
        raise ValueError("spiral_noise must be non-negative")
    rng = np.random.default_rng(random_state)
    counts = (n_samples // 2, n_samples - n_samples // 2)
    theta = np.linspace(0.2, 2.0 * np.pi * turns, counts[0])
    radius = np.linspace(0.2, 1.0, counts[0])
    first = np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))
    second = np.column_stack(
        (radius * np.cos(theta + np.pi), radius * np.sin(theta + np.pi))
    )
    X = np.vstack((first, second))
    X += rng.normal(scale=noise, size=X.shape)
    y = np.repeat((0, 1), counts)
    order = rng.permutation(n_samples)
    return X[order], y[order]


def _make_checkerboard(
    n_samples: int,
    cells: int,
    extent: float,
    label_noise: float,
    random_state: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(cells, (int, np.integer)) or isinstance(cells, bool) or cells < 2:
        raise ValueError("checkerboard_cells must be an integer of at least 2")
    if extent <= 0:
        raise ValueError("checkerboard_extent must be positive")
    if not 0 <= label_noise <= 1:
        raise ValueError("checkerboard_label_noise must be between 0 and 1")
    rng = np.random.default_rng(random_state)
    X = rng.uniform(-extent, extent, size=(n_samples, 2))
    cell_indices = np.floor((X + extent) / (2.0 * extent) * cells).astype(int)
    y = (cell_indices[:, 0] + cell_indices[:, 1]) % 2
    flips = rng.random(n_samples) < label_noise
    return X, np.where(flips, 1 - y, y)


def _make_classification_data(
    n_samples: int,
    n_classes: int,
    class_sep: float,
    flip_y: float,
    random_state: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2")
    if not 0 <= flip_y <= 1:
        raise ValueError("classification_flip_y must be between 0 and 1")
    if class_sep <= 0:
        raise ValueError("classification_class_sep must be positive")
    return sklearn_make_classification(
        n_samples=n_samples,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        class_sep=class_sep,
        flip_y=flip_y,
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
    circle_factor: float = 0.5,
    spiral_turns: float = 1.5,
    spiral_noise: float = 0.12,
    checkerboard_cells: int = 4,
    checkerboard_extent: float = 4.0,
    checkerboard_label_noise: float = 0.0,
    anisotropy: float = 3.0,
    rotation: float = 0.35,
    classification_class_sep: float = 1.0,
    classification_flip_y: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Create one of the supported two-dimensional classification datasets.

    ``n_classes`` controls the number of classes for the ``"blobs"`` dataset.
    Blob centers are placed evenly on a circle so their separation is
    controlled by ``blob_center_radius`` rather than by a random center box.
    """

    name = _canonical_dataset(dataset)
    if name == "moon":
        return make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    if name == "circles":
        if not 0 < circle_factor < 1:
            raise ValueError("circle_factor must be between 0 and 1")
        return make_circles(
            n_samples=n_samples,
            noise=noise,
            factor=circle_factor,
            random_state=random_state,
        )
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
    if name == "xor":
        return _make_xor(
            n_samples=n_samples,
            cluster_standard_deviation=blob_cluster_standard_deviation,
            random_state=random_state,
        )
    if name == "spirals":
        return _make_spirals(
            n_samples=n_samples,
            turns=spiral_turns,
            noise=spiral_noise,
            random_state=random_state,
        )
    if name == "checkerboard":
        return _make_checkerboard(
            n_samples=n_samples,
            cells=checkerboard_cells,
            extent=checkerboard_extent,
            label_noise=checkerboard_label_noise,
            random_state=random_state,
        )
    if name == "classification":
        return _make_classification_data(
            n_samples=n_samples,
            n_classes=n_classes,
            class_sep=classification_class_sep,
            flip_y=classification_flip_y,
            random_state=random_state,
        )
    if name == "anisotropic_blobs":
        X, y = _make_blobs(
            n_samples=n_samples,
            n_classes=n_classes,
            center_radius=blob_center_radius,
            cluster_standard_deviation=blob_cluster_standard_deviation,
            random_state=random_state,
        )
        if anisotropy <= 0:
            raise ValueError("anisotropy must be positive")
        transform_angle = float(rotation)
        rotation_matrix = np.array(
            [
                [np.cos(transform_angle), -np.sin(transform_angle)],
                [np.sin(transform_angle), np.cos(transform_angle)],
            ]
        )
        return X @ np.diag((anisotropy, 1.0)) @ rotation_matrix.T, y
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
        "circle_factor": 0.5,
        "spiral_turns": 1.5,
        "spiral_noise": 0.12,
        "checkerboard_cells": 4,
        "checkerboard_extent": 4.0,
        "checkerboard_label_noise": 0.0,
        "anisotropy": 3.0,
        "rotation": 0.35,
        "classification_class_sep": 1.0,
        "classification_flip_y": 0.05,
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
    model = BayesianModelAveragingClassifier(**model_options)
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
        ax.contour(
            xx,
            yy,
            probability,
            levels=[0.25, 0.75],
            linewidths=1.4,
            linestyles=":",
            colors="black",
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
        f"Bayesian model averaging on {result.dataset}\n"
        f"{result.model.n_estimators_} estimators, test accuracy = {result.test_accuracy:.3f}"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    if output_path is not None:
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig
