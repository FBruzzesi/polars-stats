from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from polars_stats import Exponential

if TYPE_CHECKING:
    import polars as pl


def test_std_is_sqrt_variance(rate: float, unit_frame: pl.DataFrame) -> None:
    e = Exponential(rate=rate)
    result = unit_frame.select(r=e.std() ** 2 - e.variance()).item(0, "r")
    assert result == pytest.approx(0.0, abs=1e-12)
