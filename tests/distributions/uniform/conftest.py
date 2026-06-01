from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from scipy.stats import uniform as _scipy_uniform

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

SEED = 42
DEFAULT_SIZE = 1000


@pytest.fixture(scope="session")
def seed() -> int:
    return SEED


@pytest.fixture
def scipy_uniform() -> Callable[[float, float], Any]:
    """Factory for a frozen `scipy.stats.uniform` reparametrised from `(min, max)`.

    Returns `Any` so callers can use the continuous-only methods (`pdf`, `logpdf`) without the
    stub picking the discrete frozen overload.
    """

    def _make(mn: float, mx: float) -> Any:  # noqa: ANN401
        return _scipy_uniform(loc=mn, scale=mx - mn)

    return _make


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=SEED)


@pytest.fixture
def frame(rng: np.random.Generator) -> Callable[..., pl.DataFrame]:
    """Frame with an `x` column and per-row bounds `lo < hi` for column-valued sampling."""

    def _make(size: int = DEFAULT_SIZE) -> pl.DataFrame:
        lo = rng.uniform(-5.0, 0.0, size=size)
        width = rng.uniform(0.5, 5.0, size=size)
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
