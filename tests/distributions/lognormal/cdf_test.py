from __future__ import annotations

import math
from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import LogNormal

if TYPE_CHECKING:
    from collections.abc import Callable


def test_cdf_is_half_at_median(params: tuple[float, float]) -> None:
    mu, sigma = params
    median = math.exp(mu)
    result = pl.DataFrame({"x": [median]}).select(r=LogNormal(mu=mu, sigma=sigma).cdf(pl.col("x")))["r"].item()
    assert result == pytest.approx(0.5, abs=1e-12)


def test_cdf_is_monotone_non_decreasing(
    params: tuple[float, float], value_grid: Callable[[float, float], list[float]]
) -> None:
    mu, sigma = params
    xs = value_grid(mu, sigma)
    result = pl.DataFrame({"x": xs}).select(r=LogNormal(mu=mu, sigma=sigma).cdf(pl.col("x")))["r"]
    assert result.is_sorted()
    assert result.is_between(0.0, 1.0).all()


def test_cdf_column_params() -> None:
    # x = exp(mu) is the median, so cdf = 0.5 regardless of sigma.
    df = pl.DataFrame({"mu": [0.0, 1.0], "sigma": [1.0, 0.5], "x": [1.0, math.e]})
    result = df.select(r=LogNormal(mu=pl.col("mu"), sigma=pl.col("sigma")).cdf(pl.col("x")))["r"]
    expected = pl.Series("r", [0.5, 0.5], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_cdf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"x": [1.0, None]}, schema={"x": pl.Float64})
    result = df.select(r=LogNormal().cdf(pl.col("x")))["r"]
    expected = pl.Series("r", [0.5, None], dtype=pl.Float64)
    assert_series_equal(result, expected)
