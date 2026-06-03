from __future__ import annotations

import math

import polars as pl
from polars.testing import assert_series_equal

from polars_stats import LogNormal


def _entropy(mu: float, sigma: float) -> float:
    # Differential entropy of a LogNormal: mu + 0.5 * log(2 * pi * e * sigma^2).
    return mu + 0.5 * math.log(2.0 * math.pi * math.e * sigma**2)


def test_entropy_is_log_form_column_params() -> None:
    mus = [0.0, -1.0, 1.0]
    sigmas = [1.0, 0.25, 0.75]
    df = pl.DataFrame({"mu": mus, "sigma": sigmas})
    result = df.select(r=LogNormal(mu=pl.col("mu"), sigma=pl.col("sigma")).entropy())["r"]
    expected = pl.Series("r", [_entropy(m, s) for m, s in zip(mus, sigmas, strict=True)], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_entropy_propagates_null_params() -> None:
    df = pl.DataFrame(
        {"mu": [0.0, None, 1.0], "sigma": [1.0, 2.0, None]}, schema={"mu": pl.Float64, "sigma": pl.Float64}
    )
    result = df.select(r=LogNormal(mu=pl.col("mu"), sigma=pl.col("sigma")).entropy())["r"]
    expected = pl.Series("r", [_entropy(0.0, 1.0), None, None], dtype=pl.Float64)
    assert_series_equal(result, expected)
