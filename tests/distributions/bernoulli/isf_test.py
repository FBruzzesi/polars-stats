from __future__ import annotations

import math

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import Bernoulli


@pytest.mark.parametrize("q", [0.1, 0.5, 0.9])
def test_isf_is_ppf_of_complement(unit_frame: pl.DataFrame, q: float) -> None:
    # isf(q) == ppf(1 - q) by base-class default.
    p = 0.3
    isf = unit_frame.select(v=Bernoulli(p=p).isf(q)).item(0, "v")
    ppf_comp = unit_frame.select(v=Bernoulli(p=p).ppf(1 - q)).item(0, "v")
    assert isf == ppf_comp


def test_isf_propagates_null_in_quantile() -> None:
    # isf(0.1) = ppf(0.9) = 1.0 for p=0.3; isf(0.9) = ppf(0.1) = 0.0.
    df = pl.DataFrame({"q": [0.1, None, 0.9]}, schema={"q": pl.Float64})
    result = df.select(v=Bernoulli(p=0.3).isf(pl.col("q")))["v"]
    expected = pl.Series("v", [1.0, None, 0.0], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_isf_propagates_nan_in_quantile(unit_frame: pl.DataFrame) -> None:
    # Same NaN contract as `ppf`: a NaN quantile propagates as NaN (scipy semantics).
    result = unit_frame.select(v=Bernoulli(p=0.3).isf(float("nan"))).item(0, "v")
    assert math.isnan(result)


def test_isf_propagates_null_in_p(p_with_null: pl.DataFrame) -> None:
    # isf(0.5) == ppf(0.5): 0.0 when p <= 0.5, 1.0 when p > 0.5, null when p is null.
    result = p_with_null.select(v=Bernoulli(p=pl.col("p")).isf(0.5))["v"]
    expected = pl.Series("v", [0.0, None, 1.0], dtype=pl.Float64)
    assert_series_equal(result, expected)
