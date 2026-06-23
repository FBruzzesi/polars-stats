from __future__ import annotations

import math

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import Exponential


def test_log_cdf_stable_in_deep_right_tail() -> None:
    # At `rate * x = 40`, `1 - exp(-rate * x)` rounds to exactly `1.0`, so the naive `log(cdf)`
    # collapses to `0.0`. The `log1p(-sf)` form keeps the true value `log(1 - exp(-40)) ~= -exp(-40)`.
    result = pl.DataFrame({"x": [40.0]}).select(r=Exponential(rate=1.0).log_cdf(pl.col("x")))["r"].item()
    assert result < 0.0  # not collapsed to 0.0 like the naive form
    assert result == pytest.approx(-math.exp(-40.0), rel=1e-6)


def test_log_cdf_is_neg_inf_at_or_below_zero(rate: float) -> None:
    # `cdf(x) = 0` for `x <= 0`, so its log is `-inf` there.
    df = pl.DataFrame({"x": [-1.0, 0.0]})
    result = df.select(r=Exponential(rate=rate).log_cdf(pl.col("x")))["r"]
    expected = pl.Series("r", [float("-inf"), float("-inf")], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_log_cdf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"x": [1.0, None]}, schema={"x": pl.Float64})
    result = df.select(r=Exponential(rate=1.0).log_cdf(pl.col("x")))["r"]
    expected = pl.Series("r", [math.log(1 - math.exp(-1.0)), None], dtype=pl.Float64)
    assert_series_equal(result, expected)
