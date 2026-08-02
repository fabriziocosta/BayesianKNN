"""Parallel execution helpers used by the estimator layer."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar

from joblib import Parallel, delayed

Item = TypeVar("Item")
Result = TypeVar("Result")


def parallel_map(
    function: Callable[[Item], Result], items: Iterable[Item], n_jobs: int | None
) -> list[Result]:
    """Apply a function in stable input order without nested worker pools."""

    return Parallel(n_jobs=n_jobs)(delayed(function)(item) for item in items)
