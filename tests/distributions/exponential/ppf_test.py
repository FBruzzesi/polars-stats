from __future__ import annotations

import math

import polars as pl

from polars_stats import Exponential
from tests._polars_compat import assert_series_equal


def test_ppf_is_cdf_inverse(rate: float) -> None:
    interior = [0.1, 0.5, 0.9]
    e = Exponential(rate=rate)
    result = pl.DataFrame({"q": interior}).select(r=e.cdf(e.ppf(pl.col("q"))))["r"]
    expected = pl.Series("r", interior, dtype=pl.Float64)
    assert_series_equal(result, expected, rel_tol=0.0, abs_tol=1e-12)


def test_ppf_endpoints(rate: float) -> None:
    # `ppf(0) = 0` (the support boundary), `ppf(1) = +inf` (the unbounded right tail).
    df = pl.DataFrame({"q": [0.0, 1.0]})
    result = df.select(r=Exponential(rate=rate).ppf(pl.col("q")))["r"]
    expected = pl.Series("r", [0.0, float("inf")], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_ppf_out_of_range_is_null() -> None:
    df = pl.DataFrame({"q": [-0.1, 0.0, 0.5, 1.0, 1.1]})
    result = df.select(r=Exponential(rate=1.0).ppf(pl.col("q")))["r"]
    expected = pl.Series("r", [None, 0.0, math.log(2.0), float("inf"), None], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_ppf_propagates_null_in_quantile() -> None:
    df = pl.DataFrame({"q": [0.2, None, 0.8]}, schema={"q": pl.Float64})
    result = df.select(r=Exponential(rate=2.0).ppf(pl.col("q")))["r"]
    expected = pl.Series("r", [-math.log(0.8) / 2, None, -math.log(0.2) / 2], dtype=pl.Float64)
    assert_series_equal(result, expected)
