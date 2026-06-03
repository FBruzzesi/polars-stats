from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_series_equal
from scipy.stats import binom as scipy_binom

from polars_stats import Binomial
from tests.distributions.binomial.conftest import N_TRIALS


@pytest.mark.parametrize("p", [0.1, 0.3, 0.5, 0.7, 0.9])
@pytest.mark.parametrize("q", [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
def test_ppf_matches_scipy_interior(p: float, q: float, unit_frame: pl.DataFrame) -> None:
    # Interior quantiles only: scipy's discrete ppf returns the below-support sentinel -1 at q == 0,
    # which this implementation (clamped to the support) does not reproduce.
    result = unit_frame.select(v=Binomial(N_TRIALS, p).ppf(q)).item(0, "v")
    assert result == pytest.approx(float(scipy_binom.ppf(q, N_TRIALS, p)))


def test_ppf_is_integer_valued() -> None:
    qs = [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]
    result = pl.DataFrame({"q": qs}).select(r=Binomial(N_TRIALS, 0.37).ppf(pl.col("q")))["r"]
    assert (result == result.floor()).all()


def test_ppf_endpoints(unit_frame: pl.DataFrame) -> None:
    # At q == 1 this matches scipy (n); at q == 0 it returns the support lower bound 0, not scipy's -1.
    assert unit_frame.select(v=Binomial(N_TRIALS, 0.4).ppf(1.0)).item(0, "v") == float(N_TRIALS)
    assert unit_frame.select(v=Binomial(N_TRIALS, 0.4).ppf(0.0)).item(0, "v") == 0.0


@pytest.mark.parametrize("q", [-0.1, 1.5])
def test_ppf_out_of_range_quantile_is_null(q: float, unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=Binomial(N_TRIALS, 0.4).ppf(q)).item(0, "v")
    assert result is None


def test_ppf_propagates_null_in_quantile() -> None:
    df = pl.DataFrame({"q": [0.1, None, 0.9]}, schema={"q": pl.Float64})
    result = df.select(v=Binomial(N_TRIALS, 0.3).ppf(pl.col("q")))["v"]
    expected = pl.Series(
        "v", [scipy_binom.ppf(0.1, N_TRIALS, 0.3), None, scipy_binom.ppf(0.9, N_TRIALS, 0.3)], dtype=pl.Float64
    )
    assert_series_equal(result, expected)


def test_ppf_propagates_null_in_params(params_with_null: pl.DataFrame) -> None:
    result = params_with_null.select(v=Binomial(pl.col("n"), pl.col("p")).ppf(0.5))["v"]
    expected = pl.Series("v", [scipy_binom.ppf(0.5, 10, 0.3), None, scipy_binom.ppf(0.5, 8, 0.8)], dtype=pl.Float64)
    assert_series_equal(result, expected)
