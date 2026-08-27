from __future__ import annotations

import math

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import Geometric


@pytest.mark.parametrize("p", [0.1, 0.25, 0.5, 0.75])
@pytest.mark.parametrize("value", [1, 2, 10])
def test_log_cdf_is_log_of_cdf(p: float, value: int, unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=Geometric(p=p).log_cdf(value)).item(0, "v")
    expected = math.log(1 - (1 - p) ** value)
    assert result == pytest.approx(expected)


@pytest.mark.parametrize("value", [-1, 0, 0.5])
def test_log_cdf_below_support_is_negative_inf(value: float, unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=Geometric(p=0.3).log_cdf(value)).item(0, "v")
    assert result == float("-inf")


def test_log_cdf_tiny_p_keeps_full_precision(unit_frame: pl.DataFrame) -> None:
    # The cdf sits a hair below 1 for tiny p, so its linear log collapses to 0.0;
    # entering through log1p(-p) keeps the whole regime: here log_cdf reads log(k * p).
    result = unit_frame.select(v=Geometric(p=1e-17).log_cdf(1000)).item(0, "v")
    assert result == pytest.approx(math.log(1000 * 1e-17), rel=1e-12)


@pytest.mark.parametrize("value", [100, 10_000, 1_000_000])
def test_log_cdf_deep_tail_does_not_underflow(value: int, unit_frame: pl.DataFrame) -> None:
    # cdf = 1 - (1-p)^k rounds to exactly 1.0 long before here, so a naive log would answer 0.
    p = 0.001
    result = unit_frame.select(v=Geometric(p=p).log_cdf(value)).item(0, "v")
    assert result == pytest.approx(math.log1p(-math.exp(value * math.log1p(-p))))


def test_log_cdf_propagates_null_in_p(p_with_null: pl.DataFrame) -> None:
    result = p_with_null.select(r=Geometric(p=pl.col("p")).log_cdf(2))["r"]
    expected = pl.Series("r", [math.log(1 - (1 - 0.3) ** 2), None, math.log(1 - (1 - 0.8) ** 2)], dtype=pl.Float64)
    assert_series_equal(result, expected)
