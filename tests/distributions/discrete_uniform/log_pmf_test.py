from __future__ import annotations

import math

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import DiscreteUniform


@pytest.mark.parametrize(
    ("lo", "hi", "value", "expected"),
    [
        (1, 6, 1.0, math.log(1 / 6)),
        (1, 6, 6.0, math.log(1 / 6)),
        (1, 6, 0.0, float("-inf")),  # off-support answers are exactly -inf, not log of a rounded 0
        (1, 6, 7.0, float("-inf")),
        (1, 6, 2.5, float("-inf")),
        (3, 3, 3.0, 0.0),  # the point mass carries all of it: log(1)
    ],
)
def test_log_pmf_scalar(lo: int, hi: int, value: float, expected: float, unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=DiscreteUniform(min=lo, max=hi).log_pmf(value)).item(0, "v")
    assert result == expected


def test_log_pmf_off_support_is_negative_inf() -> None:
    df = pl.DataFrame({"x": [-1.0, 0.5, 99.0]})
    result = df.select(r=DiscreteUniform(min=1, max=6).log_pmf(pl.col("x")))["r"]
    assert_series_equal(result, pl.Series("r", [float("-inf")] * 3, dtype=pl.Float64))


def test_log_pmf_propagates_null_in_bounds(bounds_with_null: pl.DataFrame) -> None:
    result = bounds_with_null.select(r=DiscreteUniform(min=pl.col("lo"), max=pl.col("hi")).log_pmf(2.0))["r"]
    assert result[0] == pytest.approx(math.log(1 / 6))
    assert result[1] is None
    assert result[2] is None


def test_log_pmf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"x": [1.0, None]}, schema={"x": pl.Float64})
    result = df.select(r=DiscreteUniform(min=1, max=6).log_pmf(pl.col("x")))["r"]
    assert result[0] == pytest.approx(math.log(1 / 6))
    assert result[1] is None
