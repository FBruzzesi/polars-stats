from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import Geometric

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize("p", [0.1, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize(
    ("value", "expected_fn"),
    [
        (1, lambda p: p),
        (2, lambda p: (1 - p) * p),
        (3, lambda p: (1 - p) ** 2 * p),
        (10, lambda p: (1 - p) ** 9 * p),
        (0, lambda _p: 0.0),  # below the support: the first trial is trial 1
        (-1, lambda _p: 0.0),
        (2.5, lambda _p: 0.0),  # non-integer support points
    ],
)
def test_pmf_scalar_p(
    p: float,
    value: float,
    expected_fn: Callable[[float], float],
    unit_frame: pl.DataFrame,
) -> None:
    result = unit_frame.select(v=Geometric(p=p).pmf(value)).item(0, "v")
    assert result == pytest.approx(expected_fn(p))


def test_pmf_column_p() -> None:
    probs = [0.1, 0.4, 0.6, 0.9]
    df = pl.DataFrame({"p": probs})
    pmf_at_1 = df.select(v=Geometric(p=pl.col("p")).pmf(1))["v"]
    pmf_at_2 = df.select(v=Geometric(p=pl.col("p")).pmf(2))["v"]
    assert_series_equal(pmf_at_1, pl.Series("v", probs, dtype=pl.Float64))
    assert_series_equal(pmf_at_2, pl.Series("v", [(1 - x) * x for x in probs], dtype=pl.Float64))


def test_pmf_column_value() -> None:
    p = 0.3
    df = pl.DataFrame({"v": [-1, 0, 1, 2]}, schema={"v": pl.Int64})
    result = df.select(r=Geometric(p=p).pmf(pl.col("v")))["r"]
    expected = pl.Series("r", [0.0, 0.0, p, (1 - p) * p], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_pmf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"v": [0, None, 1]}, schema={"v": pl.Int64})
    result = df.select(r=Geometric(p=0.3).pmf(pl.col("v")))["r"]
    expected = pl.Series("r", [0.0, None, 0.3], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_pmf_propagates_null_in_p(p_with_null: pl.DataFrame) -> None:
    result = p_with_null.select(r=Geometric(p=pl.col("p")).pmf(1))["r"]
    expected = pl.Series("r", [0.3, None, 0.8], dtype=pl.Float64)
    assert_series_equal(result, expected)
