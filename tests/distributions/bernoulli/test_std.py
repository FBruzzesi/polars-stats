from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from polars_stats import Bernoulli

if TYPE_CHECKING:
    import polars as pl


@pytest.mark.parametrize("p", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_std_scalar(p: float, unit_frame: pl.DataFrame) -> None:
    out = unit_frame.select(v=Bernoulli(p=p).std()).item(0, "v")
    assert out == pytest.approx(math.sqrt(p * (1 - p)))
