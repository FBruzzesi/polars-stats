from __future__ import annotations

import polars as pl

from polars_stats import Exponential
from tests._polars_compat import assert_series_equal


def test_isf_out_of_range_is_null() -> None:
    # `isf(q) = ppf(1 - q)`: `isf(0) = ppf(1) = +inf`, `isf(1) = ppf(0) = 0`.
    df = pl.DataFrame({"q": [-0.1, 0.0, 1.0, 1.1]})
    result = df.select(r=Exponential(rate=1.0).isf(pl.col("q")))["r"]
    expected = pl.Series("r", [None, float("inf"), 0.0, None], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_isf_complements_ppf(rate: float) -> None:
    qs = [0.1, 0.5, 0.9]
    e = Exponential(rate=rate)
    df = pl.DataFrame({"q": qs})
    result = df.select(r=e.isf(pl.col("q")))["r"]
    expected = df.select(r=e.ppf(1 - pl.col("q")))["r"]
    assert_series_equal(result, expected, rel_tol=0.0, abs_tol=1e-12)
