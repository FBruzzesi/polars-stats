from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import DiscreteUniform


@pytest.mark.parametrize(
    ("lo", "hi", "expected"),
    [(1, 6, 3.5), (-3, 2, -0.5), (3, 3, 3.0), (0, 7, 3.5)],
)
def test_mean(lo: int, hi: int, expected: float, unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=DiscreteUniform(min=lo, max=hi).mean()).item(0, "v")
    assert result == pytest.approx(expected)


def test_mean_can_be_a_half_integer() -> None:
    # For even N the midpoint falls between two support points; that is the answer, not a point.
    lo, hi = 1, 6
    got = pl.DataFrame({"_": [0]}).select(v=DiscreteUniform(min=lo, max=hi).mean())["v"][0]
    assert got == (lo + hi) / 2


def test_mean_propagates_null_in_bounds(bounds_with_null: pl.DataFrame) -> None:
    result = bounds_with_null.select(v=DiscreteUniform(min=pl.col("lo"), max=pl.col("hi")).mean())["v"]
    assert_series_equal(result, pl.Series("v", [3.5, None, None], dtype=pl.Float64))
