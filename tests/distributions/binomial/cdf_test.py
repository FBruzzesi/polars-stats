from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_series_equal
from scipy.stats import binom as scipy_binom

from polars_stats import Binomial
from tests.distributions.binomial.conftest import N_TRIALS


@pytest.mark.parametrize("p", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("value", [-1.0, 0.0, 2.5, 3.0, 9.0, 10.0, 11.0])
def test_cdf_matches_scipy(p: float, value: float, unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=Binomial(N_TRIALS, p).cdf(value)).item(0, "v")
    assert result == pytest.approx(float(scipy_binom.cdf(value, N_TRIALS, p)))


def test_cdf_floors_value() -> None:
    # cdf is a step function: it agrees with the value floored to the integer support.
    n, p = N_TRIALS, 0.4
    df = pl.DataFrame({"v": [2.0, 2.3, 2.99]})
    result = df.select(r=Binomial(n, p).cdf(pl.col("v")))["r"]
    expected = pl.Series("r", [scipy_binom.cdf(2, n, p)] * 3, dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_cdf_below_support_is_zero_above_is_one(unit_frame: pl.DataFrame) -> None:
    assert unit_frame.select(v=Binomial(N_TRIALS, 0.3).cdf(-0.5)).item(0, "v") == 0.0
    assert unit_frame.select(v=Binomial(N_TRIALS, 0.3).cdf(N_TRIALS)).item(0, "v") == pytest.approx(1.0)


def test_cdf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"v": [0.0, None, 10.0]}, schema={"v": pl.Float64})
    result = df.select(r=Binomial(N_TRIALS, 0.3).cdf(pl.col("v")))["r"]
    expected = pl.Series("r", [scipy_binom.cdf(0, N_TRIALS, 0.3), None, 1.0], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_cdf_propagates_null_in_params(params_with_null: pl.DataFrame) -> None:
    result = params_with_null.select(r=Binomial(pl.col("n"), pl.col("p")).cdf(3))["r"]
    expected = pl.Series("r", [scipy_binom.cdf(3, 10, 0.3), None, scipy_binom.cdf(3, 8, 0.8)], dtype=pl.Float64)
    assert_series_equal(result, expected)
