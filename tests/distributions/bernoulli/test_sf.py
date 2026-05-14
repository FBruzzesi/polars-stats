from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from polars_stats import Bernoulli

if TYPE_CHECKING:
    import polars as pl


def test_sf_is_one_minus_cdf(unit_frame: pl.DataFrame) -> None:
    p, value = 0.3, 0
    cdf = unit_frame.select(v=Bernoulli(p=p).cdf(value)).item(0, "v")
    sf = unit_frame.select(v=Bernoulli(p=p).sf(value)).item(0, "v")
    assert sf == pytest.approx(1 - cdf)
