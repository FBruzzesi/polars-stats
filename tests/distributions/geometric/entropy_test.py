from __future__ import annotations

import math

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import Geometric


@pytest.mark.parametrize("p", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_entropy_interior(p: float, unit_frame: pl.DataFrame) -> None:
    q = 1 - p
    result = unit_frame.select(v=Geometric(p=p).entropy()).item(0, "v")
    expected = (-q * math.log(q) - p * math.log(p)) / p
    assert result == pytest.approx(expected)


def test_entropy_degenerate_endpoint_is_zero(unit_frame: pl.DataFrame) -> None:
    # 0 * log 0 = 0 convention: the degenerate point mass at p = 1 has zero entropy.
    result = unit_frame.select(v=Geometric(p=1.0).entropy()).item(0, "v")
    assert result == 0.0


def test_entropy_at_half_equals_two_log_two(unit_frame: pl.DataFrame) -> None:
    half = unit_frame.select(v=Geometric(p=0.5).entropy()).item(0, "v")
    assert half == pytest.approx(2 * math.log(2))


@pytest.mark.parametrize("p", [0.1, 0.3, 0.7])
def test_entropy_grows_as_p_shrinks(p: float, unit_frame: pl.DataFrame) -> None:
    # The geometric distribution spreads out as success becomes rarer.
    smaller = unit_frame.select(v=Geometric(p=p / 2).entropy()).item(0, "v")
    larger = unit_frame.select(v=Geometric(p=p).entropy()).item(0, "v")
    assert smaller > larger


def test_entropy_propagates_null_in_p() -> None:
    # Includes the degenerate endpoint on purpose: its entropy is 0 by convention,
    # so this also guards the "0 * log 0 = 0" branch in the null-propagation path.
    df = pl.DataFrame({"p": [0.5, None, 1.0]}, schema={"p": pl.Float64})
    result = df.select(v=Geometric(p=pl.col("p")).entropy())["v"]
    expected = pl.Series("v", [2 * math.log(2), None, 0.0], dtype=pl.Float64)
    assert_series_equal(result, expected)
