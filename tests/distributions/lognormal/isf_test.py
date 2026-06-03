from __future__ import annotations

import polars as pl

from polars_stats import LogNormal
from tests._polars_compat import assert_series_equal


def test_isf_equals_ppf_of_complement(params: tuple[float, float]) -> None:
    mu, sigma = params
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    d = LogNormal(mu=mu, sigma=sigma)
    isf = pl.DataFrame({"q": qs}).select(r=d.isf(pl.col("q")))["r"]
    ppf_comp = pl.DataFrame({"q": [1 - q for q in qs]}).select(r=d.ppf(pl.col("q")))["r"]
    assert_series_equal(isf, ppf_comp, rel_tol=0.0, abs_tol=1e-12)


def test_isf_endpoints_map_to_support_boundaries() -> None:
    # isf(0) = ppf(1) = +inf, isf(1) = ppf(0) = 0.
    df = pl.DataFrame({"q": [0.0, 1.0]})
    result = df.select(r=LogNormal().isf(pl.col("q")))["r"]
    expected = pl.Series("r", [float("inf"), 0.0], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_isf_out_of_range_is_null() -> None:
    df = pl.DataFrame({"q": [-0.1, 0.0, 1.0, 1.1]})
    result = df.select(r=LogNormal().isf(pl.col("q")))["r"]
    expected = pl.Series("r", [None, float("inf"), 0.0, None], dtype=pl.Float64)
    assert_series_equal(result, expected)
