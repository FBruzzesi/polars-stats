from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import DiscreteUniform


@pytest.mark.parametrize(
    ("lo", "hi", "value", "expected"),
    [
        (1, 6, 0.5, 0.0),  # below the support
        (1, 6, 1.0, 1 / 6),
        (1, 6, 3.7, 3 / 6),  # non-integers accumulate to floor(value)
        (1, 6, 6.0, 1.0),
        (1, 6, 6.5, 1.0),
        (-3, 2, -3.0, 1 / 6),
        (-3, 2, -4.0, 0.0),
        (3, 3, 3.0, 1.0),  # the point mass: cdf(max) == cdf(min) == 1
    ],
)
def test_cdf_scalar(lo: int, hi: int, value: float, expected: float, unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=DiscreteUniform(min=lo, max=hi).cdf(value)).item(0, "v")
    assert result == pytest.approx(expected)


def test_cdf_max_is_exactly_one() -> None:
    # The visible difference from scipy's exclusive upper bound: `cdf(max)` is exactly 1, where
    # scipy's `randint.cdf(max)` answers `(max - low + 1) / (high - low) < 1`.
    for lo, hi in [(1, 6), (-5, 9), (0, 100)]:
        got = pl.DataFrame({"x": [float(hi)]}).select(r=DiscreteUniform(min=lo, max=hi).cdf("x"))["r"][0]
        assert got == 1.0


def test_cdf_column_value() -> None:
    df = pl.DataFrame({"x": [0.5, 2.0, 6.0, 9.0]})
    result = df.select(r=DiscreteUniform(min=1, max=6).cdf(pl.col("x")))["r"]
    expected = pl.Series("r", [0.0, 2 / 6, 1.0, 1.0], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_cdf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"x": [2.0, None]}, schema={"x": pl.Float64})
    result = df.select(r=DiscreteUniform(min=1, max=6).cdf(pl.col("x")))["r"]
    assert_series_equal(result, pl.Series("r", [2 / 6, None], dtype=pl.Float64))


def test_cdf_propagates_null_in_bounds(bounds_with_null: pl.DataFrame) -> None:
    result = bounds_with_null.select(r=DiscreteUniform(min=pl.col("lo"), max=pl.col("hi")).cdf(2.0))["r"]
    assert_series_equal(result, pl.Series("r", [2 / 6, None, None], dtype=pl.Float64))
