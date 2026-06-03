from __future__ import annotations

import math

import polars as pl
from polars.testing import assert_series_equal

from polars_stats import LogNormal


def test_median_is_exp_mu_column_params() -> None:
    mus = [0.0, -1.0, 1.0]
    df = pl.DataFrame({"mu": mus, "sigma": [1.0, 0.25, 0.75]})
    result = df.select(r=LogNormal(mu=pl.col("mu"), sigma=pl.col("sigma")).median())["r"]
    expected = pl.Series("r", [math.exp(m) for m in mus], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_median_propagates_null_params() -> None:
    df = pl.DataFrame(
        {"mu": [0.0, None, 1.0], "sigma": [1.0, 2.0, None]}, schema={"mu": pl.Float64, "sigma": pl.Float64}
    )
    result = df.select(r=LogNormal(mu=pl.col("mu"), sigma=pl.col("sigma")).median())["r"]
    expected = pl.Series("r", [1.0, None, None], dtype=pl.Float64)
    assert_series_equal(result, expected)
