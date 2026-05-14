from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from polars_stats import Bernoulli

if TYPE_CHECKING:
    import polars as pl


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


def test_entropy_is_maximised_at_half(unit_frame: pl.DataFrame) -> None:
    half = unit_frame.select(v=Bernoulli(p=0.5).entropy()).item(0, "v")
    for p in (0.1, 0.3, 0.7, 0.9):
        other = unit_frame.select(v=Bernoulli(p=p).entropy()).item(0, "v")
        assert half > other
    assert half == pytest.approx(math.log(2))
