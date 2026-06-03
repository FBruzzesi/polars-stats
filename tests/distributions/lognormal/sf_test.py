from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import LogNormal

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
    out = pl.DataFrame({"x": xs}).select(total=d.cdf(pl.col("x")) + d.sf(pl.col("x")))["total"]
    np.testing.assert_allclose(out.to_numpy(), 1.0, atol=1e-12, rtol=0)


def test_sf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"x": [1.0, None]}, schema={"x": pl.Float64})
    result = df.select(r=LogNormal().sf(pl.col("x")))["r"]
    expected = pl.Series("r", [0.5, None], dtype=pl.Float64)
    assert_series_equal(result, expected)
