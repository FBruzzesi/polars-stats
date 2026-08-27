from __future__ import annotations

import math

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import Geometric


@pytest.mark.parametrize("p", [0.1, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("value", [1, 2, 5, 20])
def test_log_pmf_is_log_of_pmf(p: float, value: int, unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=Geometric(p=p).log_pmf(value)).item(0, "v")
    mass = (1 - p) ** (value - 1) * p
    expected = math.log(mass) if mass > 0 else float("-inf")
    assert result == pytest.approx(expected)


@pytest.mark.parametrize("value", [-1, 0, 0.5, 2.5])
def test_log_pmf_off_support_is_negative_inf(value: float, unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=Geometric(p=0.3).log_pmf(value)).item(0, "v")
    assert result == float("-inf")


def test_log_pmf_tiny_p_keeps_full_precision(unit_frame: pl.DataFrame) -> None:
    # log(1 - p) collapses to 0.0 for tiny p; log1p(-p) keeps it, so the whole
    # small-p regime must not come back as -inf through the base pmf().log().
    result = unit_frame.select(v=Geometric(p=1e-17).log_pmf(1000)).item(0, "v")
    assert result == pytest.approx(math.log(1e-17))


def test_log_pmf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"v": [0, None, 1]}, schema={"v": pl.Int64})
    result = df.select(r=Geometric(p=0.3).log_pmf(pl.col("v")))["r"]
    expected = pl.Series("r", [float("-inf"), None, math.log(0.3)], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_log_pmf_propagates_null_in_p(p_with_null: pl.DataFrame) -> None:
    result = p_with_null.select(r=Geometric(p=pl.col("p")).log_pmf(1))["r"]
    expected = pl.Series("r", [math.log(0.3), None, math.log(0.8)], dtype=pl.Float64)
    assert_series_equal(result, expected)
