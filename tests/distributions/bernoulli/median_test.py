from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import Bernoulli


@pytest.mark.parametrize(("p", "expected"), [(0.3, False), (0.5, False), (0.7, True)])
def test_median(p: float, *, expected: bool, unit_frame: pl.DataFrame) -> None:
    # median = ppf(0.5) = (0.5 > 1 - p)
    result = unit_frame.select(v=Bernoulli(p=p).median()).item(0, "v")
    assert result is expected


def test_median_propagates_null_in_p(p_with_null: pl.DataFrame) -> None:
    result = p_with_null.select(v=Bernoulli(p=pl.col("p")).median())["v"]
    expected = pl.Series("v", [False, None, True], dtype=pl.Boolean)
    assert_series_equal(result, expected)
