from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import DiscreteUniform


@pytest.mark.parametrize(
    ("lo", "hi", "value", "expected"),
    [
        (1, 6, 1.0, 1 / 6),  # the lower bound is on the support
        (1, 6, 3.0, 1 / 6),
        (1, 6, 6.0, 1 / 6),  # so is the upper one: both bounds inclusive
        (1, 6, 0.0, 0.0),  # below
        (1, 6, 7.0, 0.0),  # above
        (1, 6, 2.5, 0.0),  # non-integers carry no mass
        (-3, 2, -3.0, 1 / 6),
        (3, 3, 3.0, 1.0),  # the min == max point mass
    ],
)
def test_pmf_scalar(lo: int, hi: int, value: float, expected: float, unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=DiscreteUniform(min=lo, max=hi).pmf(value)).item(0, "v")
    assert result == pytest.approx(expected)


def test_pmf_column_value() -> None:
    lo, hi = 1, 6
    df = pl.DataFrame({"x": [0.5, 1.0, 4.0, 6.0, 9.0]})
    result = df.select(r=DiscreteUniform(min=lo, max=hi).pmf(pl.col("x")))["r"]
    expected = pl.Series("r", [0.0] + [1 / 6] * 3 + [0.0], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_pmf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"x": [1.0, None, 6.0]}, schema={"x": pl.Float64})
    result = df.select(r=DiscreteUniform(min=1, max=6).pmf(pl.col("x")))["r"]
    assert_series_equal(result, pl.Series("r", [1 / 6, None, 1 / 6], dtype=pl.Float64))


def test_pmf_propagates_null_in_bounds(bounds_with_null: pl.DataFrame) -> None:
    # The one fully-valid row is (1, 6): mass 1/6. A null in either bound nulls the row.
    result = bounds_with_null.select(r=DiscreteUniform(min=pl.col("lo"), max=pl.col("hi")).pmf(2.0))["r"]
    assert_series_equal(result, pl.Series("r", [1 / 6, None, None], dtype=pl.Float64))
