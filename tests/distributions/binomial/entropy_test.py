from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_series_equal
from scipy.stats import binom as scipy_binom

from polars_stats import Binomial
from tests.distributions.binomial.conftest import N_TRIALS


@pytest.mark.parametrize("p", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_entropy_matches_scipy(p: float, unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=Binomial(N_TRIALS, p).entropy()).item(0, "v")
    assert result == pytest.approx(float(scipy_binom.entropy(N_TRIALS, p)))


@pytest.mark.parametrize("p", [0.0, 1.0])
def test_entropy_degenerate_endpoints_are_zero(p: float, unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=Binomial(N_TRIALS, p).entropy()).item(0, "v")
    assert result == 0.0


def test_entropy_propagates_null_in_params(params_with_null: pl.DataFrame) -> None:
    result = params_with_null.select(v=Binomial(pl.col("n"), pl.col("p")).entropy())["v"]
    expected = pl.Series("v", [scipy_binom.entropy(10, 0.3), None, scipy_binom.entropy(8, 0.8)], dtype=pl.Float64)
    assert_series_equal(result, expected)
