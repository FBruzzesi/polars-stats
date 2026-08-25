from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import Geometric


@pytest.mark.parametrize(("p", "expected"), [(0.2, 4.0), (0.3, 2.0), (0.5, 1.0), (0.9, 1.0)])
def test_median(p: float, expected: float, unit_frame: pl.DataFrame) -> None:
    # median = ppf(0.5): the smallest k whose cdf reaches one half.
    result = unit_frame.select(v=Geometric(p=p).median()).item(0, "v")
    assert result == expected


def test_median_propagates_null_in_p(p_with_null: pl.DataFrame) -> None:
    result = p_with_null.select(v=Geometric(p=pl.col("p")).median())["v"]
    expected = pl.Series("v", [2.0, None, 1.0], dtype=pl.Float64)
    assert_series_equal(result, expected)
