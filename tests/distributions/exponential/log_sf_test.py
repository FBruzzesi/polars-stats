from __future__ import annotations

import polars as pl
from polars.testing import assert_series_equal

from polars_stats import Exponential


def test_log_sf_is_zero_at_or_below_zero(rate: float) -> None:
    # `sf(x) = 1` for `x <= 0`, so its log is `0` there.
    df = pl.DataFrame({"x": [-1.0, 0.0]})
    result = df.select(r=Exponential(rate=rate).log_sf(pl.col("x")))["r"]
    expected = pl.Series("r", [0.0, 0.0], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_log_sf_matches_neg_rate_x(rate: float) -> None:
    xs = [0.0, 0.5, 1.0, 5.0]
    df = pl.DataFrame({"x": xs})
    result = df.select(r=Exponential(rate=rate).log_sf(pl.col("x")))["r"]
    expected = pl.Series("r", [-rate * x for x in xs], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_log_sf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"x": [1.0, None]}, schema={"x": pl.Float64})
    result = df.select(r=Exponential(rate=2.0).log_sf(pl.col("x")))["r"]
    expected = pl.Series("r", [-2.0, None], dtype=pl.Float64)
    assert_series_equal(result, expected)
