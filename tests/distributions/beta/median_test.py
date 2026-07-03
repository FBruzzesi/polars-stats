from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import Beta


@pytest.mark.parametrize("shape", [0.5, 1.0, 2.0, 5.0])
def test_median_is_half_for_symmetric_shapes(shape: float) -> None:
    result = pl.DataFrame({"_": [0]}).select(r=Beta(a=shape, b=shape).median()).item(0, "r")
    assert result == pytest.approx(0.5, abs=1e-9)


def test_median_equals_ppf_half_column_params() -> None:
    # No closed form: `median` is the base-class `ppf(0.5)` fallback, so the two must agree exactly.
    df = pl.DataFrame({"a": [2.0, 0.5, 5.0], "b": [3.0, 0.5, 1.0]})
    dist = Beta(a=pl.col("a"), b=pl.col("b"))
    median = df.select(r=dist.median())["r"]
    ppf_half = df.select(r=dist.ppf(0.5))["r"]
    assert_series_equal(median, ppf_half)


def test_median_propagates_null_params() -> None:
    # A null in either parameter nulls the row (null a, then null b).
    df = pl.DataFrame({"a": [1.0, None, 1.0], "b": [1.0, 2.0, None]}, schema={"a": pl.Float64, "b": pl.Float64})
    result = df.select(r=Beta(a=pl.col("a"), b=pl.col("b")).median())["r"]
    expected = pl.Series("r", [0.5, None, None], dtype=pl.Float64)
    assert_series_equal(result, expected, check_exact=False)
