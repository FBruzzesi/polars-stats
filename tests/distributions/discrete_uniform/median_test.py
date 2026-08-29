from __future__ import annotations

from fractions import Fraction

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import DiscreteUniform


@pytest.mark.parametrize(
    ("lo", "hi", "expected"),
    [
        (1, 6, 3.5),  # the midpoint, which for even N is not a support point
        (-3, 2, -0.5),
        (3, 3, 3.0),
        (1, 8, 4.5),
        (0, 7, 3.5),
    ],
)
def test_median(lo: int, hi: int, expected: float, unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=DiscreteUniform(min=lo, max=hi).median()).item(0, "v")
    assert result == expected


# scipy's `randint.median` is `ppf(0.5)` and lands on a support point; this crate reports the
# midpoint, a documented divergence from scipy.
@pytest.mark.parametrize(("lo", "hi", "midpoint", "ppf_half"), [(1, 6, 3.5, 3.0), (1, 8, 4.5, 4.0)])
def test_median_is_the_midpoint_not_ppf_half(
    lo: int, hi: int, midpoint: float, ppf_half: float, unit_frame: pl.DataFrame
) -> None:
    d = DiscreteUniform(min=lo, max=hi)
    result = unit_frame.select(median=d.median(), ppf=d.ppf(0.5))
    assert result["median"][0] == midpoint
    assert result["ppf"][0] == ppf_half


def test_median_propagates_null_in_bounds(bounds_with_null: pl.DataFrame) -> None:
    result = bounds_with_null.select(v=DiscreteUniform(min=pl.col("lo"), max=pl.col("hi")).median())["v"]
    assert_series_equal(result, pl.Series("v", [3.5, None, None], dtype=pl.Float64))


@pytest.mark.parametrize(("lo", "hi"), [(2**62, 2**62 + 10), (2**62, 2**63 - 1), (-(2**61), 2**61 + 1)])
def test_median_does_not_wrap_near_the_int64_ceiling(lo: int, hi: int, unit_frame: pl.DataFrame) -> None:
    """The same `(min + max) / 2` wrap as `mean`, which `median` computes independently."""
    result = unit_frame.select(v=DiscreteUniform(min=lo, max=hi).median()).item(0, "v")
    assert result == float(Fraction(lo + hi, 2))
