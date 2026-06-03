from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_series_equal
from scipy.stats import binom as scipy_binom

from polars_stats import Binomial
from tests.distributions.binomial.conftest import N_TRIALS


@pytest.mark.parametrize("p", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_pmf_integer_support_matches_scipy(p: float, unit_frame: pl.DataFrame) -> None:
    for k in range(N_TRIALS + 1):
        result = unit_frame.select(v=Binomial(N_TRIALS, p).pmf(k)).item(0, "v")
        assert result == pytest.approx(float(scipy_binom.pmf(k, N_TRIALS, p)))


@pytest.mark.parametrize("value", [-1.0, 2.5, 11.0, 100.0])
def test_pmf_off_support_is_zero(value: float, unit_frame: pl.DataFrame) -> None:
    # Below zero, non-integer, and above n all carry zero mass.
    result = unit_frame.select(v=Binomial(N_TRIALS, 0.3).pmf(value)).item(0, "v")
    assert result == 0.0


def test_pmf_column_value() -> None:
    n, p = N_TRIALS, 0.3
    df = pl.DataFrame({"v": [-1.0, 0.0, 2.5, 3.0, 11.0]})
    result = df.select(r=Binomial(n, p).pmf(pl.col("v")))["r"]
    expected = pl.Series("r", [0.0, scipy_binom.pmf(0, n, p), 0.0, scipy_binom.pmf(3, n, p), 0.0], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_pmf_column_n_and_p() -> None:
    df = pl.DataFrame({"n": [5, 10, 20], "p": [0.2, 0.5, 0.8]})
    result = df.select(r=Binomial(pl.col("n"), pl.col("p")).pmf(3))["r"]
    expected = pl.Series(
        "r", [scipy_binom.pmf(3, nn, pp) for nn, pp in [(5, 0.2), (10, 0.5), (20, 0.8)]], dtype=pl.Float64
    )
    assert_series_equal(result, expected)


def test_pmf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"v": [0, None, 3]}, schema={"v": pl.Int64})
    result = df.select(r=Binomial(N_TRIALS, 0.3).pmf(pl.col("v")))["r"]
    expected = pl.Series(
        "r", [scipy_binom.pmf(0, N_TRIALS, 0.3), None, scipy_binom.pmf(3, N_TRIALS, 0.3)], dtype=pl.Float64
    )
    assert_series_equal(result, expected)


def test_pmf_propagates_null_in_params(params_with_null: pl.DataFrame) -> None:
    result = params_with_null.select(r=Binomial(pl.col("n"), pl.col("p")).pmf(3))["r"]
    expected = pl.Series("r", [scipy_binom.pmf(3, 10, 0.3), None, scipy_binom.pmf(3, 8, 0.8)], dtype=pl.Float64)
    assert_series_equal(result, expected)
