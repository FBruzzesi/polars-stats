from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

SEED = 42
DEFAULT_SIZE = 1000
# Fixed trial count used by the per-method functional tests; the scipy-parity suite sweeps several.
N_TRIALS = 10


@pytest.fixture(scope="session")
def seed() -> int:
    return SEED


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=SEED)


@pytest.fixture
def frame(rng: np.random.Generator) -> Callable[..., pl.DataFrame]:
    def _make(size: int = DEFAULT_SIZE) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "x": list(range(size)),
                "n1": rng.integers(1, 50, size=size),
                "n2": rng.integers(1, 50, size=size),
                "p1": rng.uniform(0.01, 0.99, size=size),
                "p2": rng.uniform(0.01, 0.99, size=size),
            },
        )

    return _make


@pytest.fixture
def unit_frame() -> pl.DataFrame:
    """Single-row frame for evaluating scalar-output expressions."""
    return pl.DataFrame({"_": [0]})


@pytest.fixture
def params_with_null() -> pl.DataFrame:
    """`n` and `p` columns sharing a null on the middle row: (n=10, null, 8) / (p=0.3, null, 0.8)."""
    return pl.DataFrame(
        {"n": [10, None, 8], "p": [0.3, None, 0.8]},
        schema={"n": pl.Int64, "p": pl.Float64},
    )
