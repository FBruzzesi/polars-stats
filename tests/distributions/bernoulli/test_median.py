from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from polars_stats import Bernoulli

if TYPE_CHECKING:
    import polars as pl


@pytest.mark.parametrize(("p", "expected"), [(0.3, False), (0.5, False), (0.7, True)])
def test_median(p: float, *, expected: bool, unit_frame: pl.DataFrame) -> None:
    # median = ppf(0.5) = (0.5 > 1 - p)
    out = unit_frame.select(v=Bernoulli(p=p).median()).item(0, "v")
    assert out is expected
