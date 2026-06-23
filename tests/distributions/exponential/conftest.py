from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

SEED = 42
DEFAULT_SIZE = 1000

# Rates exposed through the `rate` fixture so no test file redeclares them.
PARAMS = [0.1, 0.5, 1.0, 2.0, 5.0]


@pytest.fixture(scope="session")
def seed() -> int:
    return SEED


@pytest.fixture(params=PARAMS, ids=[f"rate={r}" for r in PARAMS])
def rate(request: pytest.FixtureRequest) -> float:
    """A single rate (λ). Requesting this fixture parametrises the test over `PARAMS`."""
    return request.param  # type: ignore[no-any-return]


@pytest.fixture
def frame() -> Callable[..., pl.DataFrame]:
    """Factory for a frame with an `x` column and a per-row positive `rate` for column-valued sampling.

    Seeds a fresh generator on every call, so the data is reproducible regardless of test execution
    order, selection (`-k`) or sharding, rather than depending on a shared session-scoped stream.
    """

    def _make(size: int = DEFAULT_SIZE) -> pl.DataFrame:
        local_rng = np.random.default_rng(seed=SEED)
        rates = local_rng.uniform(0.5, 5.0, size=size)
        return pl.DataFrame({"x": list(range(size)), "rate": rates})

    return _make


@pytest.fixture
def value_grid() -> Callable[[float], list[float]]:
    """Evaluation points for a given `rate`: below support, the boundary `0`, interior, and the tail."""

    def _make(rate: float) -> list[float]:
        mean = 1.0 / rate
        return [-mean, 0.0, 0.25 * mean, mean, 2.0 * mean, 5.0 * mean]

    return _make


@pytest.fixture
def unit_frame() -> pl.DataFrame:
    """Single-row frame for evaluating scalar-output expressions."""
    return pl.DataFrame({"_": [0]})
