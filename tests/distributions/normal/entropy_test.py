from __future__ import annotations

import math

import polars as pl
from polars.testing import assert_series_equal

from polars_stats import Normal


def test_entropy_is_log_form_column_params() -> None:
    sigmas = [1.0, 0.5, 5.0]
    df = pl.DataFrame({"mu": [0.0, -3.0, 10.0], "sigma": sigmas})
    result = df.select(r=Normal(mu=pl.col("mu"), sigma=pl.col("sigma")).entropy())["r"]
    # Differential entropy of a normal: 0.5 * log(2*pi*e*sigma^2), independent of the mean.
    expected = pl.Series("r", [0.5 * math.log(2.0 * math.pi * math.e * s**2) for s in sigmas], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_entropy_propagates_null_params() -> None:
    # A null in either parameter nulls the row (null mu, then null sigma).
    df = pl.DataFrame(
        {"mu": [0.0, None, 1.0], "sigma": [1.0, 2.0, None]}, schema={"mu": pl.Float64, "sigma": pl.Float64}
    )
    result = df.select(r=Normal(mu=pl.col("mu"), sigma=pl.col("sigma")).entropy())["r"]
    expected = pl.Series("r", [0.5 * math.log(2.0 * math.pi * math.e), None, None], dtype=pl.Float64)
    assert_series_equal(result, expected)
