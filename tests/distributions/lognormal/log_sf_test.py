from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from polars.testing import assert_series_equal

from polars_stats import LogNormal

if TYPE_CHECKING:
    from collections.abc import Callable


def test_log_sf_equals_log_of_sf(value_grid: Callable[[float, float], list[float]]) -> None:
    mu, sigma = 0.5, 1.0
    xs = value_grid(mu, sigma)
    d = LogNormal(mu=mu, sigma=sigma)
    out = pl.DataFrame({"x": xs}).select(diff=d.log_sf(pl.col("x")) - d.sf(pl.col("x")).log())["diff"]
    np.testing.assert_allclose(out.to_numpy(), 0.0, atol=1e-12, rtol=0)


def test_log_sf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"x": [1.0, None]}, schema={"x": pl.Float64})
    result = df.select(r=LogNormal().log_sf(pl.col("x")))["r"]
    expected = pl.Series("r", [math.log(0.5), None], dtype=pl.Float64)
    assert_series_equal(result, expected)
