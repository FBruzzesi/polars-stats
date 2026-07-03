from __future__ import annotations

import math

import polars as pl

from polars_stats import Beta
from tests._polars_compat import assert_series_equal


def test_log_sf_equals_log_of_sf(value_grid: list[float]) -> None:
    b = Beta(a=2.0, b=3.0)
    result = pl.DataFrame({"x": value_grid}).select(diff=b.log_sf(pl.col("x")) - b.sf(pl.col("x")).log())["diff"]
    expected = pl.Series("diff", [0.0] * result.len(), dtype=pl.Float64)
    assert_series_equal(result, expected, rel_tol=0.0, abs_tol=1e-12)


def test_log_sf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"x": [0.5, None]}, schema={"x": pl.Float64})
    result = df.select(r=Beta(a=2.0, b=2.0).log_sf(pl.col("x")))["r"]
    expected = pl.Series("r", [math.log(0.5), None], dtype=pl.Float64)
    assert_series_equal(result, expected)
