from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import Beta


def test_pdf_beta_2_2_peak() -> None:
    # Beta(2, 2) density is 6x(1 - x), peaking at 1.5 in the middle of the support.
    result = pl.DataFrame({"x": [0.5]}).select(r=Beta(a=2.0, b=2.0).pdf(pl.col("x")))["r"].item()
    assert result == pytest.approx(1.5, abs=1e-12)


def test_pdf_outside_support_is_zero() -> None:
    df = pl.DataFrame({"x": [-1.0, -1e-9, 1.0 + 1e-9, 2.0]})
    result = df.select(r=Beta(a=2.0, b=3.0).pdf(pl.col("x")))["r"]
    expected = pl.Series("r", [0.0, 0.0, 0.0, 0.0], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_pdf_diverges_at_boundaries_for_shapes_below_one() -> None:
    # A shape < 1 makes the density divergent at the matching boundary (the issue caveat).
    df = pl.DataFrame({"x": [0.0, 1.0]})
    result = df.select(r=Beta(a=0.5, b=0.5).pdf(pl.col("x")))["r"]
    expected = pl.Series("r", [float("inf"), float("inf")], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_pdf_column_params() -> None:
    # Beta(2, 3) density is 12x(1 - x)^2; Beta(1, 1) is the uniform density.
    df = pl.DataFrame({"a": [2.0, 1.0], "b": [3.0, 1.0], "x": [0.5, 0.25]})
    result = df.select(r=Beta(a=pl.col("a"), b=pl.col("b")).pdf(pl.col("x")))["r"]
    expected = pl.Series("r", [1.5, 1.0], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_pdf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"x": [0.5, None, 0.25]}, schema={"x": pl.Float64})
    result = df.select(r=Beta(a=2.0, b=2.0).pdf(pl.col("x")))["r"]
    expected = pl.Series("r", [1.5, None, 1.125], dtype=pl.Float64)
    assert_series_equal(result, expected)
