from __future__ import annotations

import math

import polars as pl
import pytest

from polars_stats import Beta
from tests._polars_compat import assert_series_equal


def test_log_cdf_equals_log_of_cdf(value_grid: list[float]) -> None:
    # Central region: the stable port must agree with the naive log wherever cdf holds precision.
    b = Beta(a=2.0, b=3.0)
    result = pl.DataFrame({"x": value_grid}).select(diff=b.log_cdf(pl.col("x")) - b.cdf(pl.col("x")).log())["diff"]
    expected = pl.Series("diff", [0.0] * result.len(), dtype=pl.Float64)
    assert_series_equal(result, expected, rel_tol=0.0, abs_tol=1e-12)


def test_log_cdf_finite_in_deep_left_corner() -> None:
    # cdf underflows to 0 once x^a does; the stable log stays finite and exact. With b = 2 the
    # hypergeometric factor terminates, so the expected value is closed-form:
    # ln I_x(a, 2) = a ln x - ln a - ln B(a, 2) + ln(1 - a x / (a + 1)).
    x = 1e-3
    beta = Beta(a=200.0, b=2.0)
    frame = pl.DataFrame({"x": [x]})
    assert frame.select(r=beta.cdf(pl.col("x")))["r"].item() == 0.0
    result = frame.select(r=beta.log_cdf(pl.col("x")))["r"].item()
    expected = 200.0 * math.log(x) - math.log(200.0) + math.log(200.0 * 201.0) + math.log1p(-200.0 * x / 201.0)
    assert result == pytest.approx(expected, rel=1e-13)


def test_log_cdf_support_edges() -> None:
    # At/below 0 the cdf is 0 (log -inf); at/above 1 it is 1 (log 0), matching scipy.
    df = pl.DataFrame({"x": [-1.0, 0.0, 1.0, 1.5]})
    result = df.select(r=Beta(a=2.0, b=3.0).log_cdf(pl.col("x")))["r"]
    expected = pl.Series("r", [float("-inf"), float("-inf"), 0.0, 0.0], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_log_cdf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"x": [0.5, None]}, schema={"x": pl.Float64})
    result = df.select(r=Beta(a=2.0, b=2.0).log_cdf(pl.col("x")))["r"]
    expected = pl.Series("r", [math.log(0.5), None], dtype=pl.Float64)
    assert_series_equal(result, expected)
