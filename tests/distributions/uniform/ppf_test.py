from __future__ import annotations

import polars as pl

from polars_stats import Uniform
from tests._polars_compat import assert_series_equal


def test_ppf_is_cdf_inverse(bounds: tuple[float, float]) -> None:
    mn, mx = bounds
    interior = [0.1, 0.5, 0.9]
    u = Uniform(min=mn, max=mx)
    result = pl.DataFrame({"q": interior}).select(r=u.cdf(u.ppf(pl.col("q"))))["r"]
    expected = pl.Series("r", interior, dtype=pl.Float64)
    assert_series_equal(result, expected, rel_tol=0.0, abs_tol=1e-12)


def test_ppf_out_of_range_is_null() -> None:
    df = pl.DataFrame({"q": [-0.1, 0.0, 0.5, 1.0, 1.1]})
    result = df.select(r=Uniform(min=0.0, max=1.0).ppf(pl.col("q")))["r"]
    expected = pl.Series("r", [None, 0.0, 0.5, 1.0, None], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_ppf_propagates_null_in_quantile() -> None:
    df = pl.DataFrame({"q": [0.2, None, 0.8]}, schema={"q": pl.Float64})
    result = df.select(r=Uniform(min=0.0, max=2.0).ppf(pl.col("q")))["r"]
    expected = pl.Series("r", [0.4, None, 1.6], dtype=pl.Float64)
    assert_series_equal(result, expected)
