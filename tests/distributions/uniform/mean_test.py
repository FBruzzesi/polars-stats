from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import Uniform

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

PARAMS = [(0.0, 1.0), (-2.0, 3.0), (2.0, 5.0), (-5.0, -1.0), (0.0, 1e-3)]


@pytest.mark.parametrize(("mn", "mx"), PARAMS)
def test_mean_matches_scipy(
    mn: float,
    mx: float,
    unit_frame: pl.DataFrame,
    scipy_uniform: Callable[[float, float], Any],
) -> None:
    got = unit_frame.select(r=Uniform(min=mn, max=mx).mean()).item(0, "r")
    assert got == pytest.approx(scipy_uniform(mn, mx).mean(), abs=1e-12)


def test_mean_column_bounds() -> None:
    df = pl.DataFrame({"lo": [0.0, -2.0, 10.0], "hi": [1.0, 2.0, 20.0]})
    got = df.select(r=Uniform(min=pl.col("lo"), max=pl.col("hi")).mean())["r"]
    expected = pl.Series("r", [0.5, 0.0, 15.0], dtype=pl.Float64)
    assert_series_equal(got, expected)
