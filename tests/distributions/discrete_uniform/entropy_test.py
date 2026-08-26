from __future__ import annotations

import math

import polars as pl
import pytest

from polars_stats import DiscreteUniform


@pytest.mark.parametrize(
    ("lo", "hi", "expected"),
    [(1, 6, math.log(6)), (3, 3, 0.0), (-5, 9, math.log(15)), (0, 999, math.log(1000))],
)
def test_entropy(lo: int, hi: int, expected: float, unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=DiscreteUniform(min=lo, max=hi).entropy()).item(0, "v")
    assert result == pytest.approx(expected)


def test_entropy_of_the_point_mass_is_zero() -> None:
    got = pl.DataFrame({"_": [0]}).select(v=DiscreteUniform(min=3, max=3).entropy())["v"][0]
    assert got == 0.0


def test_entropy_propagates_null_in_bounds(bounds_with_null: pl.DataFrame) -> None:
    result = bounds_with_null.select(v=DiscreteUniform(min=pl.col("lo"), max=pl.col("hi")).entropy())["v"]
    assert result[0] == pytest.approx(math.log(6))
    assert result[1] is None
