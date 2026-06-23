from __future__ import annotations

import polars as pl
from polars.testing import assert_series_equal

from polars_stats import Exponential


def test_mean_is_inverse_rate_column() -> None:
    df = pl.DataFrame({"rate": [1.0, 2.0, 0.5]})
    result = df.select(r=Exponential(rate=pl.col("rate")).mean())["r"]
    expected = pl.Series("r", [1.0, 0.5, 2.0], dtype=pl.Float64)
    assert_series_equal(result, expected)
