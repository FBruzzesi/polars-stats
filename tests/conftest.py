"""Fixtures that deliver `tests/_registry.py` to every suite.

A test that requests `spec` runs once per distribution, and `dist` hangs off it, so no test file
repeats a builder or a parameter grid.

Fixtures compose with `pytest.mark.parametrize` but cannot be referenced from inside it, so a test
needing spec-times-method requests `spec` and parametrises the method. The `hypothesis` suite keeps its
`parametrize("spec", ALL_SPECS)` decorators: `@given` cannot draw from a fixture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from tests._registry import ALL_SPECS, CONTRACT_FRAME

if TYPE_CHECKING:
    from polars_stats.distributions._base import _UnivariateDistribution
    from tests._registry import DistSpec


@pytest.fixture(params=ALL_SPECS, ids=lambda spec: spec.name)
def spec(request: pytest.FixtureRequest) -> DistSpec:
    """One distribution's registry row. Requesting this parametrises the test over every distribution."""
    return request.param  # type: ignore[no-any-return]


@pytest.fixture
def dist(spec: DistSpec) -> _UnivariateDistribution:
    """`spec.example` as constant parameters: the validated-once fast path."""
    return spec.build("scalar")


@pytest.fixture
def contract_frame() -> pl.DataFrame:
    """`CONTRACT_FRAME`, for the `name` parameter regime. Read-only: no test mutates it."""
    return CONTRACT_FRAME


@pytest.fixture
def single_row_frame() -> pl.DataFrame:
    """Single-row frame for evaluating a scalar-output expression."""
    return pl.DataFrame({"_": [0]})
