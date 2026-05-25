from __future__ import annotations

import math

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import Bernoulli


@pytest.mark.parametrize("p", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_entropy_interior(p: float, unit_frame: pl.DataFrame) -> None:
    out = unit_frame.select(v=Bernoulli(p=p).entropy()).item(0, "v")
    expected = -p * math.log(p) - (1 - p) * math.log(1 - p)
    assert out == pytest.approx(expected)


@pytest.mark.parametrize("p", [0.0, 1.0])
def test_entropy_degenerate_endpoints_are_zero(p: float, unit_frame: pl.DataFrame) -> None:
    # 0 * log 0 = 0 convention: degenerate Bernoulli has zero entropy.
    out = unit_frame.select(v=Bernoulli(p=p).entropy()).item(0, "v")
    assert out == 0.0


def test_entropy_at_half_equals_log_two(unit_frame: pl.DataFrame) -> None:
    half = unit_frame.select(v=Bernoulli(p=0.5).entropy()).item(0, "v")
    assert half == pytest.approx(math.log(2))


@pytest.mark.parametrize("p", [0.1, 0.3, 0.7, 0.9])
def test_entropy_is_maximised_at_half(p: float, unit_frame: pl.DataFrame) -> None:
    half = unit_frame.select(v=Bernoulli(p=0.5).entropy()).item(0, "v")
    other = unit_frame.select(v=Bernoulli(p=p).entropy()).item(0, "v")
    assert half > other


def test_entropy_propagates_null_in_p() -> None:
    # Includes degenerate endpoints (0, 1) on purpose: their entropy is 0 by convention,
    # so this also guards the "0 * log 0 = 0" branch in the null-propagation path.
    df = pl.DataFrame({"p": [0.5, None, 0.0, 1.0]}, schema={"p": pl.Float64})
    result = df.select(v=Bernoulli(p=pl.col("p")).entropy())["v"]
    expected = pl.Series("v", [math.log(2), None, 0.0, 0.0], dtype=pl.Float64)
    assert_series_equal(result, expected)
