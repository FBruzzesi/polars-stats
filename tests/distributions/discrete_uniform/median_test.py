from __future__ import annotations

import math

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


def test_median_is_the_midpoint_not_ppf_half() -> None:
    # scipy's `randint.median` is `ppf(0.5)` and lands on a support point; this crate reports the
    # midpoint per the issue's convention, a documented divergence from scipy.
    lo, hi = 1, 6
    got = pl.DataFrame({"_": [0]}).select(v=DiscreteUniform(min=lo, max=hi).median())["v"][0]
    ppf = pl.DataFrame({"_": [0]}).select(v=DiscreteUniform(min=lo, max=hi).ppf(0.5))["v"][0]
    assert got == (lo + hi) / 2
    assert ppf == lo + math.ceil(0.5 * (hi - lo + 1)) - 1


def test_median_propagates_null_in_bounds(bounds_with_null: pl.DataFrame) -> None:
    result = bounds_with_null.select(v=DiscreteUniform(min=pl.col("lo"), max=pl.col("hi")).median())["v"]
    assert_series_equal(result, pl.Series("v", [3.5, None, None], dtype=pl.Float64))
