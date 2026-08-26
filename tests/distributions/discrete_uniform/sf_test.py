from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import DiscreteUniform


@pytest.mark.parametrize(
    ("lo", "hi", "value", "expected"),
    [
        (1, 6, 0.5, 1.0),
        (1, 6, 1.0, 5 / 6),
        (1, 6, 3.7, 3 / 6),
        (1, 6, 6.0, 0.0),
        (1, 6, 9.0, 0.0),
        (-3, 2, -4.0, 1.0),
        (3, 3, 3.0, 0.0),
    ],
)
def test_sf_scalar(lo: int, hi: int, value: float, expected: float, unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=DiscreteUniform(min=lo, max=hi).sf(value)).item(0, "v")
    assert result == pytest.approx(expected)


def test_sf_column_value() -> None:
    df = pl.DataFrame({"x": [0.5, 2.0, 6.0]})
    result = df.select(r=DiscreteUniform(min=1, max=6).sf(pl.col("x")))["r"]
    expected = pl.Series("r", [1.0, 4 / 6, 0.0], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_sf_propagates_null_in_bounds(bounds_with_null: pl.DataFrame) -> None:
    result = bounds_with_null.select(r=DiscreteUniform(min=pl.col("lo"), max=pl.col("hi")).sf(2.0))["r"]
    assert_series_equal(result, pl.Series("r", [4 / 6, None, None], dtype=pl.Float64))
