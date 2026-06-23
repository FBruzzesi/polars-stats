from __future__ import annotations

import math

import polars as pl
from polars.testing import assert_series_equal

from polars_stats import Exponential


def test_cdf_zero_below_support_and_rises(rate: float) -> None:
    xs = [-1.0, 0.0, 0.5, 1.0, 10.0]
    df = pl.DataFrame({"x": xs})
    result = df.select(r=Exponential(rate=rate).cdf(pl.col("x")))["r"]
    expected = pl.Series("r", [0.0 if x < 0 else 1 - math.exp(-rate * x) for x in xs], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_cdf_column_rates() -> None:
    df = pl.DataFrame({"rate": [1.0, 2.0], "x": [1.0, 0.5]})
    result = df.select(r=Exponential(rate=pl.col("rate")).cdf(pl.col("x")))["r"]
    expected = pl.Series("r", [1 - math.exp(-1.0), 1 - math.exp(-1.0)], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_cdf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"x": [0.5, None, 2.0]}, schema={"x": pl.Float64})
    result = df.select(r=Exponential(rate=1.0).cdf(pl.col("x")))["r"]
    expected = pl.Series("r", [1 - math.exp(-0.5), None, 1 - math.exp(-2.0)], dtype=pl.Float64)
    assert_series_equal(result, expected)
