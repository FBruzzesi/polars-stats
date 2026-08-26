from __future__ import annotations

import math

import polars as pl
import pytest

from polars_stats import DiscreteUniform


@pytest.mark.parametrize(
    ("lo", "hi", "value", "expected"),
    [
        (1, 6, 0.5, 0.0),  # below the support: sf == 1, log == 0
        (1, 6, 1.0, math.log(5 / 6)),
        (1, 6, 6.0, float("-inf")),  # zero mass above the inclusive max
        (1, 6, 9.0, float("-inf")),
        (-3, 2, -4.5, 0.0),
    ],
)
def test_log_sf_scalar(lo: int, hi: int, value: float, expected: float, unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=DiscreteUniform(min=lo, max=hi).log_sf(value)).item(0, "v")
    assert result == pytest.approx(expected)


def test_log_sf_propagates_null_in_bounds(bounds_with_null: pl.DataFrame) -> None:
    result = bounds_with_null.select(r=DiscreteUniform(min=pl.col("lo"), max=pl.col("hi")).log_sf(2.0))["r"]
    assert result[0] == pytest.approx(math.log(4 / 6))
    assert result[1] is None
