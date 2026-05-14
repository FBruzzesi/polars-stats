from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from polars_stats import Bernoulli

if TYPE_CHECKING:
    import polars as pl


@pytest.mark.parametrize("q", [0.1, 0.5, 0.9])
def test_isf_is_ppf_of_complement(unit_frame: pl.DataFrame, q: float) -> None:
    # isf(q) == ppf(1 - q) by base-class default.
    p = 0.3
    isf = unit_frame.select(v=Bernoulli(p=p).isf(q)).item(0, "v")
    ppf_comp = unit_frame.select(v=Bernoulli(p=p).ppf(1 - q)).item(0, "v")
    assert isf is ppf_comp
