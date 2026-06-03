from __future__ import annotations

import math

import polars as pl
import pytest
from polars.testing import assert_series_equal
from scipy.stats import binom as scipy_binom

from polars_stats import Binomial

from .conftest import N_TRIALS


@pytest.mark.parametrize("p", [0.25, 0.5, 0.75])
@pytest.mark.parametrize("value", [0.0, 3.0, 9.0])
def test_log_cdf_matches_scipy(p: float, value: float, unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=Binomial(N_TRIALS, p).log_cdf(value)).item(0, "v")
    assert result == pytest.approx(float(scipy_binom.logcdf(value, N_TRIALS, p)))


def test_log_cdf_below_support_is_neg_inf(unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=Binomial(N_TRIALS, 0.3).log_cdf(-0.5)).item(0, "v")
    assert result == -math.inf


def test_log_cdf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"v": [0.0, None, 10.0]}, schema={"v": pl.Float64})
    result = df.select(r=Binomial(N_TRIALS, 0.3).log_cdf(pl.col("v")))["r"]
    expected = pl.Series("r", [scipy_binom.logcdf(0, N_TRIALS, 0.3), None, 0.0], dtype=pl.Float64)
    assert_series_equal(result, expected)
