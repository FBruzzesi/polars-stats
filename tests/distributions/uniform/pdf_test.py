from __future__ import annotations

import polars as pl
from polars.testing import assert_series_equal

from polars_stats import Uniform


def test_pdf_constant_inside_and_zero_outside(bounds: tuple[float, float]) -> None:
    mn, mx = bounds
    width = mx - mn
    inside = [mn + 0.1 * width, (mn + mx) / 2, mn + 0.9 * width]
    outside = [mn - width, mx + width]
    df = pl.DataFrame({"x": inside + outside})
    result = df.select(r=Uniform(min=mn, max=mx).pdf(pl.col("x")))["r"]
    expected = pl.Series("r", [1 / width] * len(inside) + [0.0] * len(outside), dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_pdf_column_bounds() -> None:
    df = pl.DataFrame({"lo": [0.0, -2.0, 10.0], "hi": [1.0, 2.0, 20.0], "x": [0.5, 0.0, 25.0]})
    result = df.select(r=Uniform(min=pl.col("lo"), max=pl.col("hi")).pdf(pl.col("x")))["r"]
    expected = pl.Series("r", [1.0, 0.25, 0.0], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_pdf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"x": [0.5, None, 2.0]}, schema={"x": pl.Float64})
    result = df.select(r=Uniform(min=0.0, max=1.0).pdf(pl.col("x")))["r"]
    expected = pl.Series("r", [1.0, None, 0.0], dtype=pl.Float64)
    assert_series_equal(result, expected)
