"""Sparse random projection representation."""

from __future__ import annotations

from typing import Any

from sklearn.random_projection import SparseRandomProjection

from .base import BaseRepresentation


class SparseProjection(BaseRepresentation):
    family = "sparse"

    def __init__(self, n_components: int, random_state: int) -> None:
        self.n_components = int(n_components)
        self.random_state = int(random_state)
        self._projection = SparseRandomProjection(
            n_components=self.n_components,
            random_state=self.random_state,
        )

    def fit(self, X: Any) -> SparseProjection:
        self._projection.fit(X)
        return self

    def transform(self, X: Any) -> Any:
        return self._projection.transform(X)

    def parameters(self) -> dict[str, Any]:
        components = self._projection.components_
        return {
            "n_components": self.n_components,
            "components": components.copy(),
        }

    def sample_parameters(self, random_state: int) -> dict[str, Any]:
        return {"n_components": self.n_components, "random_state": int(random_state)}
