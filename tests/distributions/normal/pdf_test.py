from __future__ import annotations

import math

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import Normal


def test_pdf_standard_normal_peak() -> None:
    # phi(0) = 1 / sqrt(2*pi) for the standard normal.
    result = pl.DataFrame({"x": [0.0]}).select(r=Normal().pdf(pl.col("x")))["r"].item()
    assert result == pytest.approx(1.0 / math.sqrt(2.0 * math.pi), abs=1e-12)


def test_pdf_is_symmetric_about_mean() -> None:
    mean, std = 2.0, 1.5
    df = pl.DataFrame({"d": [0.5, 1.0, 4.0]})
    n = Normal(mean=mean, std_dev=std)
    left = df.select(r=n.pdf(mean - pl.col("d")))["r"]
    right = df.select(r=n.pdf(mean + pl.col("d")))["r"]
    assert_series_equal(left, right)


def test_pdf_column_params() -> None:
    df = pl.DataFrame({"mu": [0.0, 1.0], "sigma": [1.0, 2.0], "x": [0.0, 1.0]})
    result = df.select(r=Normal(mean=pl.col("mu"), std_dev=pl.col("sigma")).pdf(pl.col("x")))["r"]
    expected = pl.Series(
        "r",
        [1.0 / math.sqrt(2.0 * math.pi), 1.0 / (2.0 * math.sqrt(2.0 * math.pi))],
        dtype=pl.Float64,
    )
    assert_series_equal(result, expected)


def test_pdf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"x": [0.0, None, 1.0]}, schema={"x": pl.Float64})
    result = df.select(r=Normal().pdf(pl.col("x")))["r"]
    expected = pl.Series(
        "r",
        [1.0 / math.sqrt(2.0 * math.pi), None, math.exp(-0.5) / math.sqrt(2.0 * math.pi)],
        dtype=pl.Float64,
    )
    assert_series_equal(result, expected)
