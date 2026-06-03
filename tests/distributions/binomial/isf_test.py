from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_series_equal
from scipy.stats import binom as scipy_binom

from polars_stats import Binomial

from .conftest import N_TRIALS


@pytest.mark.parametrize("q", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_isf_is_ppf_of_complement(unit_frame: pl.DataFrame, q: float) -> None:
    p = 0.3
    isf = unit_frame.select(v=Binomial(N_TRIALS, p).isf(q)).item(0, "v")
    ppf_comp = unit_frame.select(v=Binomial(N_TRIALS, p).ppf(1 - q)).item(0, "v")
    assert isf == ppf_comp


@pytest.mark.parametrize("p", [0.3, 0.5, 0.7])
@pytest.mark.parametrize("q", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_isf_matches_scipy_interior(p: float, q: float, unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=Binomial(N_TRIALS, p).isf(q)).item(0, "v")
    assert result == pytest.approx(float(scipy_binom.isf(q, N_TRIALS, p)))


def test_isf_propagates_null_in_quantile() -> None:
    df = pl.DataFrame({"q": [0.1, None, 0.9]}, schema={"q": pl.Float64})
    result = df.select(v=Binomial(N_TRIALS, 0.3).isf(pl.col("q")))["v"]
    expected = pl.Series(
        "v", [scipy_binom.isf(0.1, N_TRIALS, 0.3), None, scipy_binom.isf(0.9, N_TRIALS, 0.3)], dtype=pl.Float64
    )
    assert_series_equal(result, expected)
