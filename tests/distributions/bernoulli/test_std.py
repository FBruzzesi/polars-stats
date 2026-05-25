from __future__ import annotations

import math

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import Bernoulli


@pytest.mark.parametrize("p", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_std_scalar(p: float, unit_frame: pl.DataFrame) -> None:
    out = unit_frame.select(v=Bernoulli(p=p).std()).item(0, "v")
    assert out == pytest.approx(math.sqrt(p * (1 - p)))


def test_std_propagates_null_in_p(p_with_null: pl.DataFrame) -> None:
    result = p_with_null.select(v=Bernoulli(p=pl.col("p")).std())["v"]
    expected = pl.Series("v", [math.sqrt(0.3 * 0.7), None, math.sqrt(0.8 * 0.2)], dtype=pl.Float64)
    assert_series_equal(result, expected)
