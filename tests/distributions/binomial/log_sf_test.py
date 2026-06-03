from __future__ import annotations

import math

import polars as pl
import pytest
from polars.testing import assert_series_equal
from scipy.stats import binom as scipy_binom

from polars_stats import Binomial
from tests.distributions.binomial.conftest import N_TRIALS


@pytest.mark.parametrize("p", [0.25, 0.5, 0.75])
@pytest.mark.parametrize("value", [0.0, 3.0, 8.0])
def test_log_sf_matches_scipy(p: float, value: float, unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=Binomial(N_TRIALS, p).log_sf(value)).item(0, "v")
    assert result == pytest.approx(float(scipy_binom.logsf(value, N_TRIALS, p)))


def test_log_sf_above_support_is_neg_inf(unit_frame: pl.DataFrame) -> None:
    # sf(n) = 0 → log_sf = -inf.
    result = unit_frame.select(v=Binomial(N_TRIALS, 0.3).log_sf(N_TRIALS)).item(0, "v")
    assert result == -math.inf


def test_log_sf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"v": [0.0, None, 3.0]}, schema={"v": pl.Float64})
    result = df.select(r=Binomial(N_TRIALS, 0.3).log_sf(pl.col("v")))["r"]
    expected = pl.Series(
        "r", [scipy_binom.logsf(0, N_TRIALS, 0.3), None, scipy_binom.logsf(3, N_TRIALS, 0.3)], dtype=pl.Float64
    )
    assert_series_equal(result, expected)
