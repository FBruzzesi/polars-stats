from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_series_equal
from scipy.stats import binom as scipy_binom

from polars_stats import Binomial

from .conftest import N_TRIALS


@pytest.mark.parametrize("p", [0.1, 0.3, 0.5, 0.75, 0.9])
def test_median_matches_scipy(p: float, unit_frame: pl.DataFrame) -> None:
    # median == ppf(0.5), which matches scipy. statrs' native floor(n*p) is a different convention
    # and is deliberately not used (it disagrees with scipy at several (n, p), e.g. n=10, p=0.75).
    result = unit_frame.select(v=Binomial(N_TRIALS, p).median()).item(0, "v")
    assert result == pytest.approx(float(scipy_binom.median(N_TRIALS, p)))


def test_median_disagrees_with_floor_np_where_scipy_does(unit_frame: pl.DataFrame) -> None:
    # Guard the convention choice: floor(10 * 0.75) = 7, but the scipy/ppf(0.5) median is 8.
    result = unit_frame.select(v=Binomial(10, 0.75).median()).item(0, "v")
    assert result == float(scipy_binom.median(10, 0.75)) == 8.0  # noqa: PLR2004


def test_median_propagates_null_in_params(params_with_null: pl.DataFrame) -> None:
    result = params_with_null.select(v=Binomial(pl.col("n"), pl.col("p")).median())["v"]
    expected = pl.Series("v", [scipy_binom.median(10, 0.3), None, scipy_binom.median(8, 0.8)], dtype=pl.Float64)
    assert_series_equal(result, expected)
