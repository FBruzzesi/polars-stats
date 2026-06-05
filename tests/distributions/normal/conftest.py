from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

SEED = 42
DEFAULT_SIZE = 1000

# `(mean, std_dev)` grid exposed through the `params` fixture so no test file redeclares it.
PARAMS = [(0.0, 1.0), (1.0, 2.0), (-3.0, 0.5), (10.0, 5.0), (0.0, 1e-3)]


@pytest.fixture(scope="session")
def seed() -> int:
    return SEED


@pytest.fixture(params=PARAMS, ids=[f"mean={m},std={s}" for m, s in PARAMS])
def params(request: pytest.FixtureRequest) -> tuple[float, float]:
    """A `(mean, std_dev)` pair. Requesting this fixture parametrises the test over `PARAMS`."""
    return request.param  # type: ignore[no-any-return]


@pytest.fixture
def frame() -> Callable[..., pl.DataFrame]:
    """Factory for a frame with an `x` column and per-row `mu` / `sigma` (`sigma > 0`) parameters.

    Seeds a fresh generator on every call, so the data is reproducible regardless of test execution
    order, selection (`-k`) or sharding, rather than depending on a shared session-scoped stream.
    """

    def _make(size: int = DEFAULT_SIZE) -> pl.DataFrame:
        local_rng = np.random.default_rng(seed=SEED)
        mu = local_rng.uniform(-5.0, 5.0, size=size)
        sigma = local_rng.uniform(0.5, 5.0, size=size)
        return pl.DataFrame({"x": list(range(size)), "mu": mu, "sigma": sigma})

    return _make


@pytest.fixture
def value_grid() -> Callable[[float, float], list[float]]:
    """Evaluation points for a `(mean, std_dev)` distribution, spanning both tails through the centre."""

    def _make(mean: float, std: float) -> list[float]:
        return [mean - 3 * std, mean - std, mean - 0.25 * std, mean, mean + 0.25 * std, mean + std, mean + 3 * std]

    return _make
