from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

SEED = 42
DEFAULT_SIZE = 1000


@pytest.fixture(scope="session")
def seed() -> int:
    return SEED


@pytest.fixture
def frame() -> Callable[..., pl.DataFrame]:
    """Factory for a frame with per-row bounds `lo <= hi` for column-valued sampling.

    Seeds a fresh generator on every call, so the data is reproducible regardless of test execution
    order, selection (`-k`) or sharding, rather than depending on a shared session-scoped stream.
    """

    def _make(size: int = DEFAULT_SIZE) -> pl.DataFrame:
        local_rng = np.random.default_rng(seed=SEED)
        lo = local_rng.integers(-20, 10, size=size)
        return pl.DataFrame({"lo": lo, "hi": lo + local_rng.integers(0, 30, size=size)})

    return _make


@pytest.fixture
def unit_frame() -> pl.DataFrame:
    """Single-row frame for evaluating scalar-output expressions."""
    return pl.DataFrame({"_": [0]})


@pytest.fixture
def bounds_with_null() -> pl.DataFrame:
    """Bounds columns with a null in the middle row."""
    return pl.DataFrame(
        {"lo": [1, None, 3], "hi": [6, 8, None]},
        schema={"lo": pl.Int64, "hi": pl.Int64},
    )
