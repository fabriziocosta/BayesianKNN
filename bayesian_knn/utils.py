"""Numerical and random-state helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.utils import check_random_state


def base_seed(random_state: Any) -> int:
    """Create one fit-level seed without sharing an RNG across workers."""

    if random_state is None:
        return int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
    if isinstance(random_state, (int, np.integer)):
        return int(random_state)
    if isinstance(random_state, np.random.Generator):
        return int(random_state.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
    legacy = check_random_state(random_state)
    return int(legacy.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32))


def child_seed(seed: int, index: int) -> int:
    """Derive a stable independent seed for a draw index."""

    sequence = np.random.SeedSequence([int(seed), int(index) & 0xFFFFFFFF])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def stable_softmax(values: np.ndarray) -> np.ndarray:
    """Return a finite normalized softmax."""

    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("softmax values must be a non-empty finite vector")
    shifted = values - np.max(values)
    weights = np.exp(shifted)
    weights /= weights.sum()
    return weights
