from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import Geometric


@pytest.mark.parametrize(
    ("p", "expected"),
    [(0.25, 12.0), (0.5, 2.0), (0.75, 4 / 9)],
)
def test_variance(p: float, expected: float, unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=Geometric(p=p).variance()).item(0, "v")
    assert result == pytest.approx(expected)


def test_variance_propagates_null_in_p(p_with_null: pl.DataFrame) -> None:
    result = p_with_null.select(v=Geometric(p=pl.col("p")).variance())["v"]
    expected = pl.Series("v", [(1 - 0.3) / 0.3**2, None, (1 - 0.8) / 0.8**2], dtype=pl.Float64)
    assert_series_equal(result, expected)
