from __future__ import annotations

import polars as pl
from polars.testing import assert_series_equal

from polars_stats import Uniform


def test_log_pdf_is_neg_inf_outside_support() -> None:
    df = pl.DataFrame({"x": [-1.0, 2.0]})
    got = df.select(r=Uniform(min=0.0, max=1.0).log_pdf(pl.col("x")))["r"]
    expected = pl.Series("r", [float("-inf"), float("-inf")], dtype=pl.Float64)
    assert_series_equal(got, expected)


def test_log_pdf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"x": [0.5, None, 2.0]}, schema={"x": pl.Float64})
    got = df.select(r=Uniform(min=0.0, max=1.0).log_pdf(pl.col("x")))["r"]
    expected = pl.Series("r", [0.0, None, float("-inf")], dtype=pl.Float64)
    assert_series_equal(got, expected)
