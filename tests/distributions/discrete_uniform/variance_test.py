from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import DiscreteUniform


@pytest.mark.parametrize(
    ("lo", "hi", "expected"),
    [(1, 6, 35 / 12), (-3, 2, 35 / 12), (3, 3, 0.0), (0, 100, (101**2 - 1) / 12)],
)
def test_variance(lo: int, hi: int, expected: float, unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=DiscreteUniform(min=lo, max=hi).variance()).item(0, "v")
    assert result == pytest.approx(expected)


def test_std() -> None:
    got = pl.DataFrame({"_": [0]}).select(v=DiscreteUniform(min=1, max=6).std())["v"][0]
    assert got == pytest.approx((35 / 12) ** 0.5)


def test_variance_propagates_null_in_bounds(bounds_with_null: pl.DataFrame) -> None:
    result = bounds_with_null.select(v=DiscreteUniform(min=pl.col("lo"), max=pl.col("hi")).variance())["v"]
    assert_series_equal(result, pl.Series("v", [35 / 12, None, None], dtype=pl.Float64))
