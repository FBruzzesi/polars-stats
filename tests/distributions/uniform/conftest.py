from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

SEED = 42
DEFAULT_SIZE = 1000

# `(min, max)` grid exposed through the `bounds` fixture so no test file redeclares it.
PARAMS = [(0.0, 1.0), (-2.0, 3.0), (2.0, 5.0), (-5.0, -1.0), (0.0, 1e-3)]


@pytest.fixture(scope="session")
def seed() -> int:
    return SEED


@pytest.fixture(params=PARAMS, ids=[f"min={mn},max={mx}" for mn, mx in PARAMS])
def bounds(request: pytest.FixtureRequest) -> tuple[float, float]:
    """A `(min, max)` parameter pair. Requesting this fixture parametrises the test over `PARAMS`."""
    return request.param  # type: ignore[no-any-return]


@pytest.fixture
def frame() -> Callable[..., pl.DataFrame]:
    """Factory for a frame with an `x` column and per-row bounds `lo < hi` for column-valued sampling.

    Seeds a fresh generator on every call, so the data is reproducible regardless of test execution
    order, selection (`-k`) or sharding, rather than depending on a shared session-scoped stream.
    """

    def _make(size: int = DEFAULT_SIZE) -> pl.DataFrame:
        local_rng = np.random.default_rng(seed=SEED)
        lo = local_rng.uniform(-5.0, 0.0, size=size)
        width = local_rng.uniform(0.5, 5.0, size=size)
        return pl.DataFrame({"x": list(range(size)), "lo": lo, "hi": lo + width})

    return _make


@pytest.fixture
def value_grid() -> Callable[[float, float], list[float]]:
    """Evaluation points for a `(min, max)` support: below, both endpoints, interior, above."""

    def _make(mn: float, mx: float) -> list[float]:
        width = mx - mn
        return [mn - width, mn, mn + 0.25 * width, (mn + mx) / 2, mn + 0.75 * width, mx, mx + width]

    return _make


@pytest.fixture
def unit_frame() -> pl.DataFrame:
    """Single-row frame for evaluating scalar-output expressions."""
    return pl.DataFrame({"_": [0]})
