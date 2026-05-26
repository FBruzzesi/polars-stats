from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import Bernoulli


def test_sf_is_one_minus_cdf(unit_frame: pl.DataFrame) -> None:
    p, value = 0.3, 0
    cdf = unit_frame.select(v=Bernoulli(p=p).cdf(value)).item(0, "v")
    sf = unit_frame.select(v=Bernoulli(p=p).sf(value)).item(0, "v")
    assert sf == pytest.approx(1 - cdf)


def test_sf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"v": [0.0, None, 1.0]}, schema={"v": pl.Float64})
    result = df.select(r=Bernoulli(p=0.3).sf(pl.col("v")))["r"]
    expected = pl.Series("r", [0.3, None, 0.0], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_sf_propagates_null_in_p(p_with_null: pl.DataFrame) -> None:
    result = p_with_null.select(r=Bernoulli(p=pl.col("p")).sf(0))["r"]
    expected = pl.Series("r", [0.3, None, 0.8], dtype=pl.Float64)
    assert_series_equal(result, expected)
