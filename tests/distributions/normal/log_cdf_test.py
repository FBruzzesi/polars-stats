from __future__ import annotations

import math
from typing import TYPE_CHECKING

import polars as pl
import pytest

from polars_stats import Normal
from tests._polars_compat import assert_series_equal

if TYPE_CHECKING:
    from collections.abc import Callable


def test_log_cdf_equals_log_of_cdf(value_grid: Callable[[float, float], list[float]]) -> None:
    mean, std = 1.0, 2.0
    xs = value_grid(mean, std)
    n = Normal(mu=mean, sigma=std)
    result = pl.DataFrame({"x": xs}).select(diff=n.log_cdf(pl.col("x")) - n.cdf(pl.col("x")).log())["diff"]
    expected = pl.Series("diff", [0.0] * result.len(), dtype=pl.Float64)
    assert_series_equal(result, expected, rel_tol=0.0, abs_tol=1e-12)


def test_log_cdf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"x": [0.0, None]}, schema={"x": pl.Float64})
    result = df.select(r=Normal().log_cdf(pl.col("x")))["r"]
    expected = pl.Series("r", [math.log(0.5), None], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_log_cdf_finite_in_deep_tail() -> None:
    # `cdf(-40)` underflows to `0`, so the naive `cdf().log()` is `-inf`; the native `ln_erfc` form
    # stays finite. Reference value is `scipy.stats.norm.logcdf(-40)`.
    df = pl.DataFrame({"x": [-40.0]})
    result = df.select(r=Normal().log_cdf(pl.col("x")))["r"]
    naive = df.select(r=Normal().cdf(pl.col("x")).log())["r"]
    assert naive.item(0) == float("-inf")
    assert result.item(0) == pytest.approx(-804.608442, rel=1e-6)
