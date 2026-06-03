from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_series_equal
from scipy.stats import binom as scipy_binom

from polars_stats import Binomial

from .conftest import N_TRIALS


@pytest.mark.parametrize("p", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("value", [-1.0, 0.0, 2.5, 3.0, 9.0, 10.0, 11.0])
def test_sf_matches_scipy(p: float, value: float, unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=Binomial(N_TRIALS, p).sf(value)).item(0, "v")
    assert result == pytest.approx(float(scipy_binom.sf(value, N_TRIALS, p)))


def test_sf_is_one_minus_cdf(unit_frame: pl.DataFrame) -> None:
    n, p, value = N_TRIALS, 0.3, 3
    cdf = unit_frame.select(v=Binomial(n, p).cdf(value)).item(0, "v")
    sf = unit_frame.select(v=Binomial(n, p).sf(value)).item(0, "v")
    assert sf == pytest.approx(1 - cdf)


def test_sf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"v": [0.0, None, 10.0]}, schema={"v": pl.Float64})
    result = df.select(r=Binomial(N_TRIALS, 0.3).sf(pl.col("v")))["r"]
    expected = pl.Series(
        "r", [scipy_binom.sf(0, N_TRIALS, 0.3), None, scipy_binom.sf(10, N_TRIALS, 0.3)], dtype=pl.Float64
    )
    assert_series_equal(result, expected)


def test_sf_propagates_null_in_params(params_with_null: pl.DataFrame) -> None:
    result = params_with_null.select(r=Binomial(pl.col("n"), pl.col("p")).sf(3))["r"]
    expected = pl.Series("r", [scipy_binom.sf(3, 10, 0.3), None, scipy_binom.sf(3, 8, 0.8)], dtype=pl.Float64)
    assert_series_equal(result, expected)
