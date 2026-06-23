from __future__ import annotations

import math

import polars as pl
from polars.testing import assert_series_equal

from polars_stats import Exponential


def test_median_is_log2_over_rate_column() -> None:
    df = pl.DataFrame({"rate": [1.0, 2.0]})
    result = df.select(r=Exponential(rate=pl.col("rate")).median())["r"]
    expected = pl.Series("r", [math.log(2.0), math.log(2.0) / 2], dtype=pl.Float64)
    assert_series_equal(result, expected)
