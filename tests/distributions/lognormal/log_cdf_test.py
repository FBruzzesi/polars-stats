from __future__ import annotations

import math
from typing import TYPE_CHECKING

import polars as pl
import pytest

from polars_stats import LogNormal
from tests._polars_compat import assert_series_equal

if TYPE_CHECKING:
    from collections.abc import Callable


def test_log_cdf_equals_log_of_cdf(value_grid: Callable[[float, float], list[float]]) -> None:
    mu, sigma = 0.5, 1.0
    xs = value_grid(mu, sigma)
    d = LogNormal(mu=mu, sigma=sigma)
    result = pl.DataFrame({"x": xs}).select(diff=d.log_cdf(pl.col("x")) - d.cdf(pl.col("x")).log())["diff"]
    expected = pl.Series("diff", [0.0] * result.len(), dtype=pl.Float64)
    assert_series_equal(result, expected, rel_tol=0.0, abs_tol=1e-12)


def test_log_cdf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"x": [1.0, None]}, schema={"x": pl.Float64})
    result = df.select(r=LogNormal().log_cdf(pl.col("x")))["r"]
    expected = pl.Series("r", [math.log(0.5), None], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_log_cdf_finite_in_deep_tail() -> None:
    # For x = exp(-40) (far-left tail), cdf underflows to 0 (naive cdf().log() is -inf); the
    # underlying-normal ln_erfc form stays finite and equals scipy.stats.norm.logcdf(-40).
    df = pl.DataFrame({"x": [math.exp(-40.0)]})
    result = df.select(r=LogNormal().log_cdf(pl.col("x")))["r"]
    naive = df.select(r=LogNormal().cdf(pl.col("x")).log())["r"]
    assert naive.item(0) == float("-inf")
    assert result.item(0) == pytest.approx(-804.608442, rel=1e-6)
