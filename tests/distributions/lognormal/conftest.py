from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

SEED = 42
DEFAULT_SIZE = 1000

# `(mu, sigma)` grid exposed through the `params` fixture so no test file redeclares it. `sigma` is
# kept moderate: the LogNormal mean/variance grow exponentially in `sigma` and lose absolute
# precision against scipy past `sigma > 5` (see the issue caveat).
PARAMS = [(0.0, 1.0), (0.5, 0.5), (-1.0, 0.25), (1.0, 0.75), (0.0, 0.1)]


@pytest.fixture(scope="session")
def seed() -> int:
    return SEED


@pytest.fixture(params=PARAMS, ids=[f"mu={m},sigma={s}" for m, s in PARAMS])
def params(request: pytest.FixtureRequest) -> tuple[float, float]:
    """A `(mu, sigma)` pair. Requesting this fixture parametrises the test over `PARAMS`."""
    return request.param  # type: ignore[no-any-return]


@pytest.fixture
def frame() -> Callable[..., pl.DataFrame]:
    """Factory for a frame with an `x` column and per-row `mu` / `sigma` (`sigma > 0`) parameters.

    Seeds a fresh generator on every call, so the data is reproducible regardless of test execution
    order, selection (`-k`) or sharding, rather than depending on a shared session-scoped stream.
    """

    def _make(size: int = DEFAULT_SIZE) -> pl.DataFrame:
        local_rng = np.random.default_rng(seed=SEED)
        mu = local_rng.uniform(-2.0, 2.0, size=size)
        sigma = local_rng.uniform(0.25, 2.0, size=size)
        return pl.DataFrame({"x": list(range(size)), "mu": mu, "sigma": sigma})

    return _make


@pytest.fixture
def value_grid() -> Callable[[float, float], list[float]]:
    """Evaluation points on the positive support of a `(mu, sigma)` LogNormal, spanning both tails.

    Geometric around the median `exp(mu)`, so the grid tracks the distribution's scale.
    """

    def _make(mu: float, sigma: float) -> list[float]:
        ks = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
        return np.exp(mu + ks * sigma).tolist()

    return _make
