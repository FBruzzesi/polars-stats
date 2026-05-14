from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from polars_stats import Bernoulli

if TYPE_CHECKING:
    import polars as pl


@pytest.mark.parametrize("p", [0.25, 0.5, 0.75])
def test_log_pmf_matches_log_of_pmf(p: float, unit_frame: pl.DataFrame) -> None:
    for value, expected in [(0, math.log(1 - p)), (1, math.log(p))]:
        out = unit_frame.select(v=Bernoulli(p=p).log_pmf(value)).item(0, "v")
        assert out == pytest.approx(expected)


def test_log_pmf_outside_support_is_neg_inf(unit_frame: pl.DataFrame) -> None:
    out = unit_frame.select(v=Bernoulli(p=0.5).log_pmf(2)).item(0, "v")
    assert math.isinf(out)
    assert out < 0
