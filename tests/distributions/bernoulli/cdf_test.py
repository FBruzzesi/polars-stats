from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import Bernoulli

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize("p", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize(
    ("value", "expected_fn"),
    [
        (-0.5, lambda _p: 0.0),
        (0, lambda p: 1 - p),
        (0.5, lambda p: 1 - p),
        (1, lambda _p: 1.0),
        (1.5, lambda _p: 1.0),
    ],
)
def test_cdf_scalar_p(
    p: float,
    value: float,
    expected_fn: Callable[[float], float],
    unit_frame: pl.DataFrame,
) -> None:
    result = unit_frame.select(v=Bernoulli(p=p).cdf(value)).item(0, "v")
    assert result == pytest.approx(expected_fn(p))


def test_cdf_column_p() -> None:
    probs = [0.1, 0.4, 0.6, 0.9]
    df = pl.DataFrame({"p": probs})
    result = df.select(v=Bernoulli(p=pl.col("p")).cdf(0))["v"]
    expected = pl.Series("v", [1 - x for x in probs], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_cdf_column_value() -> None:
    p = 0.3
    df = pl.DataFrame({"v": [-1.0, 0.0, 0.5, 1.0, 1.5]})
    result = df.select(r=Bernoulli(p=p).cdf(pl.col("v")))["r"]
    expected = pl.Series("r", [0.0, 1 - p, 1 - p, 1.0, 1.0], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_cdf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"v": [0.0, None, 1.0]}, schema={"v": pl.Float64})
    result = df.select(r=Bernoulli(p=0.3).cdf(pl.col("v")))["r"]
    expected = pl.Series("r", [0.7, None, 1.0], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_cdf_propagates_null_in_p(p_with_null: pl.DataFrame) -> None:
    result = p_with_null.select(r=Bernoulli(p=pl.col("p")).cdf(0))["r"]
    expected = pl.Series("r", [0.7, None, 1 - 0.8], dtype=pl.Float64)
    assert_series_equal(result, expected)
