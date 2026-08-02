"""Representation interfaces and factory."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseRepresentation(ABC):
    """Small interface shared by all representation families."""

    family: str

    @abstractmethod
    def fit(self, X: Any) -> "BaseRepresentation":
        raise NotImplementedError

    @abstractmethod
    def transform(self, X: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        raise NotImplementedError

    def fit_transform(self, X: Any) -> Any:
        return self.fit(X).transform(X)


def make_representation(family: str, n_components: int, random_state: int) -> BaseRepresentation:
    """Construct a configured representation without exposing it to predictors."""

    if family == "gaussian":
        from .gaussian_projection import GaussianProjection

        return GaussianProjection(n_components=n_components, random_state=random_state)
    if family == "sparse":
        from .sparse_projection import SparseProjection

        return SparseProjection(n_components=n_components, random_state=random_state)
    if family == "identity":
        from .identity import IdentityRepresentation

        return IdentityRepresentation()
    raise ValueError("representation must be 'gaussian', 'sparse', or 'identity'")
