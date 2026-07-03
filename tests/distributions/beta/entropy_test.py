from __future__ import annotations

import polars as pl
import pytest
from scipy.stats import beta as scipy_beta

from polars_stats import Beta
from tests._polars_compat import assert_series_equal


def test_entropy_matches_scipy_column_params() -> None:
    shapes = [(2.0, 3.0), (0.5, 0.5), (5.0, 1.0)]
    df = pl.DataFrame({"a": [s[0] for s in shapes], "b": [s[1] for s in shapes]})
    result = df.select(r=Beta(a=pl.col("a"), b=pl.col("b")).entropy())["r"]
    expected = pl.Series("r", [float(scipy_beta.entropy(a, b)) for a, b in shapes], dtype=pl.Float64)
    assert_series_equal(result, expected, rel_tol=0.0, abs_tol=1e-12)


def test_entropy_uniform_shapes_is_zero() -> None:
    # Beta(1, 1) is Uniform(0, 1), whose differential entropy is log(1) = 0.
    result = pl.DataFrame({"_": [0]}).select(r=Beta(a=1.0, b=1.0).entropy()).item(0, "r")
    assert result == pytest.approx(0.0, abs=1e-12)


def test_entropy_propagates_null_params() -> None:
    # A null in either parameter nulls the row (null a, then null b).
    df = pl.DataFrame({"a": [2.0, None, 1.0], "b": [3.0, 2.0, None]}, schema={"a": pl.Float64, "b": pl.Float64})
    result = df.select(r=Beta(a=pl.col("a"), b=pl.col("b")).entropy())["r"]
    expected = pl.Series("r", [float(scipy_beta.entropy(2.0, 3.0)), None, None], dtype=pl.Float64)
    assert_series_equal(result, expected, rel_tol=0.0, abs_tol=1e-12)
