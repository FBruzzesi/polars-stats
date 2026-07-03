from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

SEED = 42
DEFAULT_SIZE = 1000

# `(a, b)` shape grid exposed through the `params` fixture so no test file redeclares it. Covers the
# qualitatively distinct regimes: unimodal, U-shaped (both shapes < 1), uniform (1, 1), J-shaped, and skewed.
PARAMS = [(2.0, 3.0), (0.5, 0.5), (1.0, 1.0), (5.0, 1.0), (2.0, 8.0)]


@pytest.fixture(scope="session")
def seed() -> int:
    return SEED


@pytest.fixture(params=PARAMS, ids=[f"a={a},b={b}" for a, b in PARAMS])
def params(request: pytest.FixtureRequest) -> tuple[float, float]:
    """An `(a, b)` shape pair. Requesting this fixture parametrises the test over `PARAMS`."""
    return request.param  # type: ignore[no-any-return]


@pytest.fixture
def frame() -> Callable[..., pl.DataFrame]:
    """Factory for a frame with an `x` column and per-row positive `a` / `b` shape columns.

    Seeds a fresh generator on every call, so the data is reproducible regardless of test execution
    order, selection (`-k`) or sharding, rather than depending on a shared session-scoped stream.
    """

    def _make(size: int = DEFAULT_SIZE) -> pl.DataFrame:
        local_rng = np.random.default_rng(seed=SEED)
        a = local_rng.uniform(0.5, 5.0, size=size)
        b = local_rng.uniform(0.5, 5.0, size=size)
        return pl.DataFrame({"x": list(range(size)), "a": a, "b": b})

    return _make


@pytest.fixture
def value_grid() -> list[float]:
    """Interior evaluation points of the fixed `[0, 1]` support, spanning both tails through the centre."""
    return [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
