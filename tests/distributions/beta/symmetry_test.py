"""`X ~ Beta(a, b)` implies `1 - X ~ Beta(b, a)`, so the cdf must satisfy `F_{a,b}(x) = 1 - F_{b,a}(1 - x)`."""

from __future__ import annotations

import polars as pl

from polars_stats import Beta
from tests._polars_compat import assert_series_equal


def test_cdf_reflection_symmetry(params: tuple[float, float], value_grid: list[float]) -> None:
    a, b = params
    df = pl.DataFrame({"x": value_grid})
    lhs = df.select(r=Beta(a=a, b=b).cdf(pl.col("x")))["r"]
    rhs = df.select(r=1 - Beta(a=b, b=a).cdf(1 - pl.col("x")))["r"]
    assert_series_equal(lhs, rhs, rel_tol=0.0, abs_tol=1e-12)
