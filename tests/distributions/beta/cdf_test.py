from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import Beta


@pytest.mark.parametrize("shape", [0.5, 1.0, 2.0, 5.0])
def test_cdf_is_half_at_centre_for_symmetric_shapes(shape: float) -> None:
    result = pl.DataFrame({"x": [0.5]}).select(r=Beta(a=shape, b=shape).cdf(pl.col("x")))["r"].item()
    assert result == pytest.approx(0.5, abs=1e-12)


def test_cdf_is_monotone_non_decreasing(params: tuple[float, float], value_grid: list[float]) -> None:
    a, b = params
    result = pl.DataFrame({"x": value_grid}).select(r=Beta(a=a, b=b).cdf(pl.col("x")))["r"]
    assert result.is_sorted()
    assert result.is_between(0.0, 1.0).all()


def test_cdf_clamps_outside_support() -> None:
    df = pl.DataFrame({"x": [-0.5, 0.0, 1.0, 1.5]})
    result = df.select(r=Beta(a=2.0, b=3.0).cdf(pl.col("x")))["r"]
    expected = pl.Series("r", [0.0, 0.0, 1.0, 1.0], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_cdf_column_params() -> None:
    # Beta(1, 1) cdf is the identity on the support; Beta(2, 2) cdf is 3x^2 - 2x^3.
    df = pl.DataFrame({"a": [1.0, 2.0], "b": [1.0, 2.0], "x": [0.25, 0.5]})
    result = df.select(r=Beta(a=pl.col("a"), b=pl.col("b")).cdf(pl.col("x")))["r"]
    expected = pl.Series("r", [0.25, 0.5], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_cdf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"x": [0.5, None]}, schema={"x": pl.Float64})
    result = df.select(r=Beta(a=2.0, b=2.0).cdf(pl.col("x")))["r"]
    expected = pl.Series("r", [0.5, None], dtype=pl.Float64)
    assert_series_equal(result, expected)
