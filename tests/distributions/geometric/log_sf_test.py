from __future__ import annotations

import math

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import Geometric


@pytest.mark.parametrize("p", [0.1, 0.25, 0.5, 0.75])
@pytest.mark.parametrize("value", [1, 2, 10])
def test_log_sf_is_log_of_sf(p: float, value: int, unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=Geometric(p=p).log_sf(value)).item(0, "v")
    expected = (value) * math.log1p(-p)
    assert result == pytest.approx(expected)


@pytest.mark.parametrize("value", [-1, 0, 0.5])
def test_log_sf_below_support_is_zero(value: float, unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=Geometric(p=0.3).log_sf(value)).item(0, "v")
    assert result == 0.0


@pytest.mark.parametrize("k", [10_000, 100_000, 1_000_000])
def test_log_sf_deep_tail_does_not_underflow(k: int, unit_frame: pl.DataFrame) -> None:
    # sf = (1-p)^k underflows to exactly 0.0 around k ~ 70_000 for p = 0.001,
    # so the base sf().log() would answer -inf where this still has digits.
    p = 0.001
    result = unit_frame.select(v=Geometric(p=p).log_sf(k)).item(0, "v")
    assert result == pytest.approx(k * math.log1p(-p))


def test_log_sf_propagates_null_in_p(p_with_null: pl.DataFrame) -> None:
    result = p_with_null.select(r=Geometric(p=pl.col("p")).log_sf(2))["r"]
    expected = pl.Series("r", [2 * math.log1p(-0.3), None, 2 * math.log1p(-0.8)], dtype=pl.Float64)
    assert_series_equal(result, expected)
