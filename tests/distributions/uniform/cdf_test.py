from __future__ import annotations

import polars as pl
from polars.testing import assert_series_equal

from polars_stats import Uniform


def test_cdf_clamps_outside_support() -> None:
    df = pl.DataFrame({"x": [-10.0, 0.0, 0.5, 1.0, 10.0]})
    got = df.select(r=Uniform(min=0.0, max=1.0).cdf(pl.col("x")))["r"]
    expected = pl.Series("r", [0.0, 0.0, 0.5, 1.0, 1.0], dtype=pl.Float64)
    assert_series_equal(got, expected)


def test_cdf_column_bounds() -> None:
    df = pl.DataFrame({"lo": [0.0, -2.0], "hi": [1.0, 2.0], "x": [0.5, 0.0]})
    got = df.select(r=Uniform(min=pl.col("lo"), max=pl.col("hi")).cdf(pl.col("x")))["r"]
    expected = pl.Series("r", [0.5, 0.5], dtype=pl.Float64)
    assert_series_equal(got, expected)


def test_cdf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"x": [0.25, None, 0.75]}, schema={"x": pl.Float64})
    got = df.select(r=Uniform(min=0.0, max=1.0).cdf(pl.col("x")))["r"]
    expected = pl.Series("r", [0.25, None, 0.75], dtype=pl.Float64)
    assert_series_equal(got, expected)
