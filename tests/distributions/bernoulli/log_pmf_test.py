from __future__ import annotations

import math
from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import Bernoulli

if TYPE_CHECKING:
    from collections.abc import Callable


def _log(x: float) -> float:
    # log(0) is -inf by IEEE-754 floating-point convention; math.log raises instead, so wrap it.
    return -math.inf if x == 0.0 else math.log(x)


@pytest.mark.parametrize("p", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize(
    ("value", "expected_fn"),
    [
        (0, lambda p: _log(1 - p)),
        (1, _log),
        (-1, lambda _p: -math.inf),  # outside support → pmf = 0
        (2, lambda _p: -math.inf),
        (0.5, lambda _p: -math.inf),  # non-integer support point
    ],
)
def test_log_pmf_scalar_p(
    p: float,
    value: float,
    expected_fn: Callable[[float], float],
    unit_frame: pl.DataFrame,
) -> None:
    result = unit_frame.select(v=Bernoulli(p=p).log_pmf(value)).item(0, "v")
    assert result == pytest.approx(expected_fn(p))


def test_log_pmf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"v": [0, None, 1]}, schema={"v": pl.Int64})
    result = df.select(r=Bernoulli(p=0.3).log_pmf(pl.col("v")))["r"]
    expected = pl.Series("r", [math.log(0.7), None, math.log(0.3)], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_log_pmf_propagates_null_in_p(p_with_null: pl.DataFrame) -> None:
    result = p_with_null.select(r=Bernoulli(p=pl.col("p")).log_pmf(1))["r"]
    expected = pl.Series("r", [math.log(0.3), None, math.log(0.8)], dtype=pl.Float64)
    assert_series_equal(result, expected)
