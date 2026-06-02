from __future__ import annotations

import polars as pl
from polars.testing import assert_series_equal

from polars_stats import Normal


def test_std_equals_scale_column_params() -> None:
    df = pl.DataFrame({"mu": [0.0, -3.0, 10.0], "sigma": [1.0, 0.5, 5.0]})
    result = df.select(r=Normal(mean=pl.col("mu"), std_dev=pl.col("sigma")).std())["r"]
    expected = pl.Series("r", [1.0, 0.5, 5.0], dtype=pl.Float64)
    assert_series_equal(result, expected)
