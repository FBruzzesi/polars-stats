from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from polars.testing import assert_series_equal

from polars_stats import Normal

if TYPE_CHECKING:
    from collections.abc import Callable


def test_log_sf_equals_log_of_sf(value_grid: Callable[[float, float], list[float]]) -> None:
    mean, std = 1.0, 2.0
    xs = value_grid(mean, std)
    n = Normal(mean=mean, std_dev=std)
    out = pl.DataFrame({"x": xs}).select(diff=n.log_sf(pl.col("x")) - n.sf(pl.col("x")).log())["diff"]
    np.testing.assert_allclose(out.to_numpy(), 0.0, atol=1e-12, rtol=0)


def test_log_sf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"x": [0.0, None]}, schema={"x": pl.Float64})
    result = df.select(r=Normal().log_sf(pl.col("x")))["r"]
    expected = pl.Series("r", [float(np.log(0.5)), None], dtype=pl.Float64)
    assert_series_equal(result, expected)
