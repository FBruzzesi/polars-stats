from __future__ import annotations

import math
from typing import TYPE_CHECKING

import polars as pl
import pytest

from polars_stats import LogNormal
from tests._polars_compat import assert_series_equal

if TYPE_CHECKING:
    from collections.abc import Callable


def test_sf_is_half_at_median(params: tuple[float, float]) -> None:
    mu, sigma = params
    median = math.exp(mu)
    result = pl.DataFrame({"x": [median]}).select(r=LogNormal(mu=mu, sigma=sigma).sf(pl.col("x")))["r"].item()
    assert result == pytest.approx(0.5, abs=1e-12)


def test_sf_complements_cdf(params: tuple[float, float], value_grid: Callable[[float, float], list[float]]) -> None:
    mu, sigma = params
    xs = value_grid(mu, sigma)
    d = LogNormal(mu=mu, sigma=sigma)
    result = pl.DataFrame({"x": xs}).select(total=d.cdf(pl.col("x")) + d.sf(pl.col("x")))["total"]
    expected = pl.Series("total", [1.0] * result.len(), dtype=pl.Float64)
    assert_series_equal(result, expected, rel_tol=0.0, abs_tol=1e-12)


def test_sf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"x": [1.0, None]}, schema={"x": pl.Float64})
    result = df.select(r=LogNormal().sf(pl.col("x")))["r"]
    expected = pl.Series("r", [0.5, None], dtype=pl.Float64)
    assert_series_equal(result, expected)
