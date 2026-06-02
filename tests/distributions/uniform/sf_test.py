from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from polars.testing import assert_series_equal

from polars_stats import Uniform

if TYPE_CHECKING:
    from collections.abc import Callable


def test_sf_clamps_outside_support() -> None:
    df = pl.DataFrame({"x": [-10.0, 0.0, 0.5, 1.0, 10.0]})
    result = df.select(r=Uniform(min=0.0, max=1.0).sf(pl.col("x")))["r"]
    expected = pl.Series("r", [1.0, 1.0, 0.5, 0.0, 0.0], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_sf_complements_cdf(bounds: tuple[float, float], value_grid: Callable[[float, float], list[float]]) -> None:
    mn, mx = bounds
    xs = value_grid(mn, mx)
    u = Uniform(min=mn, max=mx)
    out = pl.DataFrame({"x": xs}).select(total=u.cdf(pl.col("x")) + u.sf(pl.col("x")))["total"]
    np.testing.assert_allclose(out.to_numpy(), 1.0, atol=1e-12, rtol=0)


def test_sf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"x": [0.25, None, 0.75]}, schema={"x": pl.Float64})
    result = df.select(r=Uniform(min=0.0, max=1.0).sf(pl.col("x")))["r"]
    expected = pl.Series("r", [0.75, None, 0.25], dtype=pl.Float64)
    assert_series_equal(result, expected)
