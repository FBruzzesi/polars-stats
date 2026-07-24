from __future__ import annotations

import polars as pl

from polars_stats import Normal
from tests._polars_compat import assert_series_equal


def test_isf_equals_ppf_of_complement(params: tuple[float, float]) -> None:
    mean, std = params
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    n = Normal(mu=mean, sigma=std)
    isf = pl.DataFrame({"q": qs}).select(r=n.isf(pl.col("q")))["r"]
    ppf_comp = pl.DataFrame({"q": [1 - q for q in qs]}).select(r=n.ppf(pl.col("q")))["r"]
    assert_series_equal(isf, ppf_comp, rel_tol=0.0, abs_tol=1e-12)


def test_isf_endpoints_are_infinite() -> None:
    df = pl.DataFrame({"q": [0.0, 1.0]})
    result = df.select(r=Normal().isf(pl.col("q")))["r"]
    expected = pl.Series("r", [float("inf"), float("-inf")], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_isf_out_of_range_is_null() -> None:
    df = pl.DataFrame({"q": [-0.1, 0.0, 1.0, 1.1]})
    result = df.select(r=Normal().isf(pl.col("q")))["r"]
    expected = pl.Series("r", [None, float("inf"), float("-inf"), None], dtype=pl.Float64)
    assert_series_equal(result, expected)
