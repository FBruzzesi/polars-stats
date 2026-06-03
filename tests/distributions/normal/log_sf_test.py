from __future__ import annotations

import math
from typing import TYPE_CHECKING

import polars as pl

from polars_stats import Normal
from tests._polars_compat import assert_series_equal

if TYPE_CHECKING:
    from collections.abc import Callable


def test_log_sf_equals_log_of_sf(value_grid: Callable[[float, float], list[float]]) -> None:
    mean, std = 1.0, 2.0
    xs = value_grid(mean, std)
    n = Normal(mean=mean, std_dev=std)
    result = pl.DataFrame({"x": xs}).select(diff=n.log_sf(pl.col("x")) - n.sf(pl.col("x")).log())["diff"]
    expected = pl.Series("diff", [0.0] * result.len(), dtype=pl.Float64)
    assert_series_equal(result, expected, rel_tol=0.0, abs_tol=1e-12)


def test_log_sf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"x": [0.0, None]}, schema={"x": pl.Float64})
    result = df.select(r=Normal().log_sf(pl.col("x")))["r"]
    expected = pl.Series("r", [math.log(0.5), None], dtype=pl.Float64)
    assert_series_equal(result, expected)
