"""`Binomial(n=1, p)` is the Bernoulli(`p`) distribution; the shared methods must agree."""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import Bernoulli, Binomial

_PROBS = [0.0, 0.2, 0.5, 0.8, 1.0]
_VALUE_GRID = [-1.0, 0.0, 0.5, 1.0, 2.0]
_QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]


@pytest.mark.parametrize("p", _PROBS)
@pytest.mark.parametrize("method", ["pmf", "log_pmf", "cdf", "log_cdf", "sf", "log_sf"])
def test_value_keyed_methods_match_bernoulli(p: float, method: str) -> None:
    df = pl.DataFrame({"x": _VALUE_GRID})
    binom = df.select(r=getattr(Binomial(1, p), method)(pl.col("x")))["r"]
    bern = df.select(r=getattr(Bernoulli(p), method)(pl.col("x")))["r"]
    assert_series_equal(binom, bern, check_names=False)


@pytest.mark.parametrize("p", _PROBS)
@pytest.mark.parametrize("method", ["mean", "variance", "std", "entropy"])
def test_scalar_methods_match_bernoulli(p: float, method: str, unit_frame: pl.DataFrame) -> None:
    binom = unit_frame.select(r=getattr(Binomial(1, p), method)()).item(0, "r")
    bern = unit_frame.select(r=getattr(Bernoulli(p), method)()).item(0, "r")
    assert binom == pytest.approx(bern)


@pytest.mark.parametrize("p", [0.2, 0.5, 0.8])
@pytest.mark.parametrize("q", _QUANTILES)
def test_ppf_matches_bernoulli(p: float, q: float, unit_frame: pl.DataFrame) -> None:
    binom = unit_frame.select(r=Binomial(1, p).ppf(q)).item(0, "r")
    bern = unit_frame.select(r=Bernoulli(p).ppf(q)).item(0, "r")
    assert binom == bern
