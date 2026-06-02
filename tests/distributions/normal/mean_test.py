from __future__ import annotations

import polars as pl
from polars.testing import assert_series_equal

from polars_stats import Normal


def test_mean_equals_location_column_params() -> None:
    df = pl.DataFrame({"mu": [0.0, -3.0, 10.0], "sigma": [1.0, 0.5, 5.0]})
    result = df.select(r=Normal(mean=pl.col("mu"), std_dev=pl.col("sigma")).mean())["r"]
    expected = pl.Series("r", [0.0, -3.0, 10.0], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_mean_propagates_null_params() -> None:
    # A null in either parameter nulls the row (null mean, then null std_dev).
    df = pl.DataFrame(
        {"mu": [0.0, None, 1.0], "sigma": [1.0, 2.0, None]}, schema={"mu": pl.Float64, "sigma": pl.Float64}
    )
    result = df.select(r=Normal(mean=pl.col("mu"), std_dev=pl.col("sigma")).mean())["r"]
    expected = pl.Series("r", [0.0, None, None], dtype=pl.Float64)
    assert_series_equal(result, expected)
