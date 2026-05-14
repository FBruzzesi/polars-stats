from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from polars_stats import Bernoulli

if TYPE_CHECKING:
    import polars as pl


def test_log_sf_matches_log_of_sf(unit_frame: pl.DataFrame) -> None:
    p, value = 0.3, 0
    out = unit_frame.select(v=Bernoulli(p=p).log_sf(value)).item(0, "v")
    assert out == pytest.approx(math.log(p))
