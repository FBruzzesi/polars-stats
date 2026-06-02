from __future__ import annotations

import polars as pl
from polars.testing import assert_series_equal

from polars_stats import Uniform


def test_isf_out_of_range_is_null() -> None:
    df = pl.DataFrame({"q": [-0.1, 0.0, 1.0, 1.1]})
    result = df.select(r=Uniform(min=0.0, max=1.0).isf(pl.col("q")))["r"]
    expected = pl.Series("r", [None, 1.0, 0.0, None], dtype=pl.Float64)
    assert_series_equal(result, expected)
