from __future__ import annotations

import math

import polars as pl
from polars.testing import assert_series_equal

from polars_stats import Exponential


def test_pdf_density_on_support_and_zero_below(rate: float) -> None:
    xs = [-1.0, 0.0, 0.5, 1.0, 2.0]
    df = pl.DataFrame({"x": xs})
    result = df.select(r=Exponential(rate=rate).pdf(pl.col("x")))["r"]
    expected = pl.Series(
        "r",
        [0.0 if x < 0 else rate * math.exp(-rate * x) for x in xs],
        dtype=pl.Float64,
    )
    assert_series_equal(result, expected)


def test_pdf_at_zero_is_rate(rate: float) -> None:
    # The density peaks at the support boundary: `pdf(0) = rate`.
    result = pl.DataFrame({"x": [0.0]}).select(r=Exponential(rate=rate).pdf(pl.col("x")))["r"]
    assert_series_equal(result, pl.Series("r", [rate], dtype=pl.Float64))


def test_pdf_column_rates() -> None:
    df = pl.DataFrame({"rate": [1.0, 2.0, 0.5], "x": [0.0, 1.0, -1.0]})
    result = df.select(r=Exponential(rate=pl.col("rate")).pdf(pl.col("x")))["r"]
    expected = pl.Series("r", [1.0, 2.0 * math.exp(-2.0), 0.0], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_pdf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"x": [0.5, None, -1.0]}, schema={"x": pl.Float64})
    result = df.select(r=Exponential(rate=1.0).pdf(pl.col("x")))["r"]
    expected = pl.Series("r", [math.exp(-0.5), None, 0.0], dtype=pl.Float64)
    assert_series_equal(result, expected)
