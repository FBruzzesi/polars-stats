from __future__ import annotations

import math

import polars as pl
import pytest
from polars.testing import assert_series_equal
from scipy.stats import binom as scipy_binom

from polars_stats import Binomial
from tests.distributions.binomial.conftest import N_TRIALS


@pytest.mark.parametrize("p", [0.25, 0.5, 0.75])
def test_log_pmf_integer_support_matches_scipy(p: float, unit_frame: pl.DataFrame) -> None:
    for k in range(N_TRIALS + 1):
        result = unit_frame.select(v=Binomial(N_TRIALS, p).log_pmf(k)).item(0, "v")
        assert result == pytest.approx(float(scipy_binom.logpmf(k, N_TRIALS, p)))


@pytest.mark.parametrize("value", [-1.0, 2.5, 11.0])
def test_log_pmf_off_support_is_neg_inf(value: float, unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=Binomial(N_TRIALS, 0.3).log_pmf(value)).item(0, "v")
    assert result == -math.inf


def test_log_pmf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"v": [0, None, 3]}, schema={"v": pl.Int64})
    result = df.select(r=Binomial(N_TRIALS, 0.3).log_pmf(pl.col("v")))["r"]
    expected = pl.Series(
        "r", [scipy_binom.logpmf(0, N_TRIALS, 0.3), None, scipy_binom.logpmf(3, N_TRIALS, 0.3)], dtype=pl.Float64
    )
    assert_series_equal(result, expected)


def test_log_pmf_propagates_null_in_params(params_with_null: pl.DataFrame) -> None:
    result = params_with_null.select(r=Binomial(pl.col("n"), pl.col("p")).log_pmf(3))["r"]
    expected = pl.Series("r", [scipy_binom.logpmf(3, 10, 0.3), None, scipy_binom.logpmf(3, 8, 0.8)], dtype=pl.Float64)
    assert_series_equal(result, expected)
