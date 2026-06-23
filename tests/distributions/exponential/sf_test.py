from __future__ import annotations

import math
from typing import TYPE_CHECKING

import polars as pl

from polars_stats import Exponential
from tests._polars_compat import assert_series_equal

if TYPE_CHECKING:
    from collections.abc import Callable


def test_sf_one_below_support_and_decays(rate: float) -> None:
    xs = [-1.0, 0.0, 0.5, 1.0, 10.0]
    df = pl.DataFrame({"x": xs})
    result = df.select(r=Exponential(rate=rate).sf(pl.col("x")))["r"]
    expected = pl.Series("r", [1.0 if x < 0 else math.exp(-rate * x) for x in xs], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_sf_complements_cdf(rate: float, value_grid: Callable[[float], list[float]]) -> None:
    xs = value_grid(rate)
    e = Exponential(rate=rate)
    result = pl.DataFrame({"x": xs}).select(total=e.cdf(pl.col("x")) + e.sf(pl.col("x")))["total"]
    expected = pl.Series("total", [1.0] * len(xs), dtype=pl.Float64)
    assert_series_equal(result, expected, rel_tol=0.0, abs_tol=1e-12)


def test_sf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"x": [0.5, None, 2.0]}, schema={"x": pl.Float64})
    result = df.select(r=Exponential(rate=1.0).sf(pl.col("x")))["r"]
    expected = pl.Series("r", [math.exp(-0.5), None, math.exp(-2.0)], dtype=pl.Float64)
    assert_series_equal(result, expected)
