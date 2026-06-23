from __future__ import annotations

import math

import polars as pl
from polars.testing import assert_series_equal

from polars_stats import Exponential


def test_entropy_is_one_minus_log_rate_column() -> None:
    df = pl.DataFrame({"rate": [1.0, 2.0, 0.5]})
    result = df.select(r=Exponential(rate=pl.col("rate")).entropy())["r"]
    expected = pl.Series("r", [1 - math.log(r) for r in (1.0, 2.0, 0.5)], dtype=pl.Float64)
    assert_series_equal(result, expected)
