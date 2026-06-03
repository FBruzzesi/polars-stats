from __future__ import annotations

import math

import polars as pl
from polars.testing import assert_series_equal

from polars_stats import LogNormal


def test_mean_is_exp_formula_column_params() -> None:
    mus = [0.0, -1.0, 1.0]
    sigmas = [1.0, 0.25, 0.75]
    df = pl.DataFrame({"mu": mus, "sigma": sigmas})
    result = df.select(r=LogNormal(mu=pl.col("mu"), sigma=pl.col("sigma")).mean())["r"]
    expected = pl.Series("r", [math.exp(m + s**2 / 2.0) for m, s in zip(mus, sigmas, strict=True)], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_mean_propagates_null_params() -> None:
    df = pl.DataFrame(
        {"mu": [0.0, None, 1.0], "sigma": [1.0, 2.0, None]}, schema={"mu": pl.Float64, "sigma": pl.Float64}
    )
    result = df.select(r=LogNormal(mu=pl.col("mu"), sigma=pl.col("sigma")).mean())["r"]
    expected = pl.Series("r", [math.exp(0.5), None, None], dtype=pl.Float64)
    assert_series_equal(result, expected)
