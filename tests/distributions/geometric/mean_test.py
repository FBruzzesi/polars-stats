from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import Geometric


@pytest.mark.parametrize(
    ("p", "expected"),
    [(0.25, 4.0), (0.3, 10 / 3), (0.5, 2.0), (1.0, 1.0)],
)
def test_mean(p: float, expected: float, unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=Geometric(p=p).mean()).item(0, "v")
    assert result == pytest.approx(expected)


def test_mean_propagates_null_in_p(p_with_null: pl.DataFrame) -> None:
    result = p_with_null.select(v=Geometric(p=pl.col("p")).mean())["v"]
    expected = pl.Series("v", [1 / 0.3, None, 1 / 0.8], dtype=pl.Float64)
    assert_series_equal(result, expected)
