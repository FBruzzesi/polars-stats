from __future__ import annotations

import polars as pl
from polars.testing import assert_series_equal

from polars_stats import Uniform


def test_variance_column_bounds() -> None:
    df = pl.DataFrame({"lo": [0.0, -2.0], "hi": [1.0, 4.0]})
    got = df.select(r=Uniform(min=pl.col("lo"), max=pl.col("hi")).variance())["r"]
    expected = pl.Series("r", [1 / 12, 36 / 12], dtype=pl.Float64)
    assert_series_equal(got, expected)
