from __future__ import annotations

import math
from typing import TYPE_CHECKING

import polars as pl

from polars_stats import LogNormal
from tests._polars_compat import assert_series_equal

if TYPE_CHECKING:
    from collections.abc import Callable


def test_log_pdf_equals_log_of_pdf(value_grid: Callable[[float, float], list[float]]) -> None:
    mu, sigma = 0.5, 1.0
    xs = value_grid(mu, sigma)
    d = LogNormal(mu=mu, sigma=sigma)
    result = pl.DataFrame({"x": xs}).select(diff=d.log_pdf(pl.col("x")) - d.pdf(pl.col("x")).log())["diff"]
    expected = pl.Series("diff", [0.0] * result.len(), dtype=pl.Float64)
    assert_series_equal(result, expected, rel_tol=0.0, abs_tol=1e-12)


def test_log_pdf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"x": [1.0, None]}, schema={"x": pl.Float64})
    result = df.select(r=LogNormal().log_pdf(pl.col("x")))["r"]
    expected = pl.Series("r", [math.log(1.0 / math.sqrt(2.0 * math.pi)), None], dtype=pl.Float64)
    assert_series_equal(result, expected)
