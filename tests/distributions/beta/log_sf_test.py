from __future__ import annotations

import math

import polars as pl
import pytest

from polars_stats import Beta
from tests._polars_compat import assert_series_equal


def test_log_sf_equals_log_of_sf(value_grid: list[float]) -> None:
    # Central region: the stable port must agree with the naive log wherever sf holds precision.
    b = Beta(a=2.0, b=3.0)
    result = pl.DataFrame({"x": value_grid}).select(diff=b.log_sf(pl.col("x")) - b.sf(pl.col("x")).log())["diff"]
    expected = pl.Series("diff", [0.0] * result.len(), dtype=pl.Float64)
    assert_series_equal(result, expected, rel_tol=0.0, abs_tol=1e-12)


def test_log_sf_finite_in_deep_right_corner() -> None:
    # sf underflows to 0 once (1 - x)^b does; the stable log stays finite and exact. Mirror of the
    # log_cdf corner test through sf(x) = I_{1-x}(b, a), with the complement taken exactly as the
    # implementation recomputes it from the rounded input.
    x = 1.0 - 1e-3
    y = 1.0 - x
    beta = Beta(a=2.0, b=200.0)
    frame = pl.DataFrame({"x": [x]})
    assert frame.select(r=beta.sf(pl.col("x")))["r"].item() == 0.0
    result = frame.select(r=beta.log_sf(pl.col("x")))["r"].item()
    expected = 200.0 * math.log(y) - math.log(200.0) + math.log(200.0 * 201.0) + math.log1p(-200.0 * y / 201.0)
    assert result == pytest.approx(expected, rel=1e-13)


def test_log_sf_support_edges() -> None:
    # At/below 0 the sf is 1 (log 0); at/above 1 it is 0 (log -inf), matching scipy.
    df = pl.DataFrame({"x": [-1.0, 0.0, 1.0, 1.5]})
    result = df.select(r=Beta(a=2.0, b=3.0).log_sf(pl.col("x")))["r"]
    expected = pl.Series("r", [0.0, 0.0, float("-inf"), float("-inf")], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_log_sf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"x": [0.5, None]}, schema={"x": pl.Float64})
    result = df.select(r=Beta(a=2.0, b=2.0).log_sf(pl.col("x")))["r"]
    expected = pl.Series("r", [math.log(0.5), None], dtype=pl.Float64)
    assert_series_equal(result, expected)
