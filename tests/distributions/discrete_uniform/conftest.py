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


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=SEED)


@pytest.fixture
def frame(rng: np.random.Generator) -> Callable[..., pl.DataFrame]:
    def _make(size: int = DEFAULT_SIZE) -> pl.DataFrame:
        lo = rng.integers(-20, 10, size=size)
        hi = lo + rng.integers(0, 30, size=size)
        return pl.DataFrame(
            {
                "lo": lo,
                "hi": hi,
                # Evaluation points spanning below `min`, inside the support (integers and not),
                # and above `max`.
                "x": rng.uniform(-25.0, 45.0, size=size),
                "q": rng.uniform(0.0, 1.0, size=size),
            },
        )

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
