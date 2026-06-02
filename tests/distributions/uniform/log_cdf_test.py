from __future__ import annotations

import numpy as np
import polars as pl
from polars.testing import assert_series_equal

from polars_stats import Uniform


def test_log_cdf_is_neg_inf_at_or_below_min() -> None:
    df = pl.DataFrame({"x": [-1.0, 0.0]})
    result = df.select(r=Uniform(min=0.0, max=1.0).log_cdf(pl.col("x")))["r"]
    expected = pl.Series("r", [float("-inf"), float("-inf")], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_log_cdf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"x": [0.5, None]}, schema={"x": pl.Float64})
    result = df.select(r=Uniform(min=0.0, max=1.0).log_cdf(pl.col("x")))["r"]
    expected = pl.Series("r", [float(np.log(0.5)), None], dtype=pl.Float64)
    assert_series_equal(result, expected)
