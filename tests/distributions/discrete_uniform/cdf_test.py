from __future__ import annotations

from fractions import Fraction

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


# `N = 49, 98, 103, 107` are the first four support counts where `N * (1 / N) != 1.0`, so these pairs
# fail if `cdf(max)` is a clamped product rather than the explicit `1.0` branch.
@pytest.mark.parametrize(("lo", "hi"), [(1, 6), (-5, 9), (0, 100), (3, 3), (0, 48), (0, 97), (0, 102), (0, 106)])
def test_cdf_max_is_exactly_one(lo: int, hi: int) -> None:
    """`cdf(max)` is exactly 1, the visible difference from scipy's exclusive upper bound."""
    got = pl.DataFrame({"x": [float(hi)]}).select(r=DiscreteUniform(min=lo, max=hi).cdf("x"))["r"][0]
    assert got == 1.0


def test_cdf_max_is_exactly_one_for_every_support_count_up_to_4000() -> None:
    """The column-routed sweep behind the curated pairs, covering all 483 counts that lose the product."""
    counts = range(1, 4001)
    df = pl.DataFrame(
        {"lo": [0] * len(counts), "hi": [c - 1 for c in counts], "x": [float(c - 1) for c in counts]},
        schema_overrides={"lo": pl.Int64, "hi": pl.Int64},
    )
    result = df.select(r=DiscreteUniform(min=pl.col("lo"), max=pl.col("hi")).cdf("x"))["r"]
    assert_series_equal(result, pl.Series("r", [1.0] * len(counts), dtype=pl.Float64))


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


def test_cdf_saturates_to_one_for_an_int_value_past_int64_half(unit_frame: pl.DataFrame) -> None:
    """The `value >= max` branch discards a wrapped count, so the `int` and `float` spellings agree."""
    d = DiscreteUniform(min=0, max=10)
    point = 2**63 - 1
    from_int = unit_frame.select(v=d.cdf(point)).item(0, "v")
    from_float = unit_frame.select(v=d.cdf(float(point))).item(0, "v")
    assert from_int == 1.0
    assert from_int == from_float


@pytest.mark.parametrize("offset", [0, 1, 5, 9, 10])
def test_cdf_is_exact_for_an_int_value_in_a_narrow_support_past_the_float_grid(
    offset: int, unit_frame: pl.DataFrame
) -> None:
    """An `int` evaluation point resolves every support point where a `Float64` one cannot."""
    lo = 2**62
    hi = lo + 10
    n = hi - lo + 1
    result = unit_frame.select(v=DiscreteUniform(min=lo, max=hi).cdf(lo + offset)).item(0, "v")
    assert result == pytest.approx(float(Fraction(offset + 1, n)), rel=1e-15)
