from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from polars_stats import Uniform

if TYPE_CHECKING:
    import polars as pl


def test_std_is_sqrt_variance(bounds: tuple[float, float], unit_frame: pl.DataFrame) -> None:
    mn, mx = bounds
    u = Uniform(min=mn, max=mx)
    result = unit_frame.select(r=u.std() ** 2 - u.variance()).item(0, "r")
    assert result == pytest.approx(0.0, abs=1e-12)
