from __future__ import annotations

import math

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import LogNormal


def test_pdf_standard_lognormal_at_one() -> None:
    # For LogNormal(0, 1): pdf(1) = 1 / (1 * sqrt(2*pi) * 1) = 1 / sqrt(2*pi).
    result = pl.DataFrame({"x": [1.0]}).select(r=LogNormal().pdf(pl.col("x")))["r"].item()
    assert result == pytest.approx(1.0 / math.sqrt(2.0 * math.pi), abs=1e-12)


def test_pdf_column_params() -> None:
    df = pl.DataFrame({"mu": [0.0, 0.0], "sigma": [1.0, 0.5], "x": [1.0, 1.0]})
    result = df.select(r=LogNormal(mu=pl.col("mu"), sigma=pl.col("sigma")).pdf(pl.col("x")))["r"]
    # At x = exp(mu) = 1 the density is 1 / (sigma * sqrt(2*pi)).
    expected = pl.Series(
        "r",
        [1.0 / math.sqrt(2.0 * math.pi), 1.0 / (0.5 * math.sqrt(2.0 * math.pi))],
        dtype=pl.Float64,
    )
    assert_series_equal(result, expected)


def test_pdf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"x": [1.0, None, math.e]}, schema={"x": pl.Float64})
    result = df.select(r=LogNormal().pdf(pl.col("x")))["r"]
    expected = pl.Series(
        "r",
        [1.0 / math.sqrt(2.0 * math.pi), None, math.exp(-0.5) / (math.e * math.sqrt(2.0 * math.pi))],
        dtype=pl.Float64,
    )
    assert_series_equal(result, expected)
