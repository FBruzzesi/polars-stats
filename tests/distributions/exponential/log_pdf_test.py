from __future__ import annotations

import math

import polars as pl
from polars.testing import assert_series_equal

from polars_stats import Exponential


def test_log_pdf_is_neg_inf_below_support(rate: float) -> None:
    df = pl.DataFrame({"x": [-1.0, -0.5]})
    result = df.select(r=Exponential(rate=rate).log_pdf(pl.col("x")))["r"]
    expected = pl.Series("r", [float("-inf"), float("-inf")], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_log_pdf_matches_log_rate_minus_rate_x(rate: float) -> None:
    xs = [0.0, 0.5, 1.0, 2.0]
    df = pl.DataFrame({"x": xs})
    result = df.select(r=Exponential(rate=rate).log_pdf(pl.col("x")))["r"]
    expected = pl.Series("r", [math.log(rate) - rate * x for x in xs], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_log_pdf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"x": [0.5, None, -1.0]}, schema={"x": pl.Float64})
    result = df.select(r=Exponential(rate=1.0).log_pdf(pl.col("x")))["r"]
    expected = pl.Series("r", [-0.5, None, float("-inf")], dtype=pl.Float64)
    assert_series_equal(result, expected)
