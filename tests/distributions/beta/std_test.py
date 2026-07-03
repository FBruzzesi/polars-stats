from __future__ import annotations

import math

import polars as pl
from polars.testing import assert_series_equal

from polars_stats import Beta


def test_std_is_sqrt_of_variance_column_params() -> None:
    df = pl.DataFrame({"a": [2.0, 1.0], "b": [3.0, 1.0]})
    result = df.select(r=Beta(a=pl.col("a"), b=pl.col("b")).std())["r"]
    expected = pl.Series("r", [0.2, math.sqrt(1.0 / 12.0)], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_std_propagates_null_params() -> None:
    # A null in either parameter nulls the row (null a, then null b).
    df = pl.DataFrame({"a": [2.0, None, 1.0], "b": [3.0, 2.0, None]}, schema={"a": pl.Float64, "b": pl.Float64})
    result = df.select(r=Beta(a=pl.col("a"), b=pl.col("b")).std())["r"]
    expected = pl.Series("r", [0.2, None, None], dtype=pl.Float64)
    assert_series_equal(result, expected)
