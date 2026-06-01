from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from polars_stats import Uniform

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    import polars as pl

PARAMS = [(0.0, 1.0), (-2.0, 3.0), (2.0, 5.0), (-5.0, -1.0), (0.0, 1e-3)]


@pytest.mark.parametrize(("mn", "mx"), PARAMS)
def test_std_matches_scipy(
    mn: float,
    mx: float,
    unit_frame: pl.DataFrame,
    scipy_uniform: Callable[[float, float], Any],
) -> None:
    got = unit_frame.select(r=Uniform(min=mn, max=mx).std()).item(0, "r")
    assert got == pytest.approx(scipy_uniform(mn, mx).std(), abs=1e-12)


@pytest.mark.parametrize(("mn", "mx"), PARAMS)
def test_std_is_sqrt_variance(mn: float, mx: float, unit_frame: pl.DataFrame) -> None:
    u = Uniform(min=mn, max=mx)
    out = unit_frame.select(r=u.std() ** 2 - u.variance()).item(0, "r")
    assert out == pytest.approx(0.0, abs=1e-12)
