from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import Geometric

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize("p", [0.1, 0.3, 0.5, 0.8])
@pytest.mark.parametrize(
    ("value", "expected_fn"),
    [
        (0.99, lambda _p: 0.0),  # below the support
        (-1.0, lambda _p: 0.0),
        (1, lambda p: p),
        (2, lambda p: 1 - (1 - p) ** 2),
        (3, lambda p: 1 - (1 - p) ** 3),
        (10, lambda p: 1 - (1 - p) ** 10),
    ],
)
def test_cdf_scalar_p(
    p: float,
    value: float,
    expected_fn: Callable[[float], float],
    unit_frame: pl.DataFrame,
) -> None:
    result = unit_frame.select(v=Geometric(p=p).cdf(value)).item(0, "v")
    assert result == pytest.approx(expected_fn(p))


def test_cdf_column_value() -> None:
    p = 0.3
    df = pl.DataFrame({"v": [-1, 0, 1, 2]}, schema={"v": pl.Int64})
    result = df.select(r=Geometric(p=p).cdf(pl.col("v")))["r"]
    expected = pl.Series("r", [0.0, 0.0, p, 1 - (1 - p) ** 2], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_cdf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"v": [0, None, 2]}, schema={"v": pl.Int64})
    result = df.select(r=Geometric(p=0.5).cdf(pl.col("v")))["r"]
    expected = pl.Series("r", [0.0, None, 0.75], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_cdf_propagates_null_in_p(p_with_null: pl.DataFrame) -> None:
    result = p_with_null.select(r=Geometric(p=pl.col("p")).cdf(2))["r"]
    expected = pl.Series("r", [0.51, None, 0.96], dtype=pl.Float64)
    assert_series_equal(result, expected)
