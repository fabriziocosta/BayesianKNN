"""Identity representation."""

from __future__ import annotations

from typing import Any

from .base import BaseRepresentation


class IdentityRepresentation(BaseRepresentation):
    family = "identity"

    def fit(self, X: Any) -> IdentityRepresentation:
        if getattr(X, "ndim", 2) != 2:
            raise ValueError("X must be two-dimensional")
        self.n_features_in_ = int(X.shape[1])
        return self

    def transform(self, X: Any) -> Any:
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the fitted data")
        return X

    def parameters(self) -> dict[str, Any]:
        return {"n_components": self.n_features_in_}

    def sample_parameters(self, random_state: int) -> dict[str, Any]:
        return {"random_state": int(random_state)}
