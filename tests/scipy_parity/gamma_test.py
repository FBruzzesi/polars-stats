from __future__ import annotations

from typing import Literal

import polars as pl
import pytest
from scipy.stats import gamma as scipy_gamma

from polars_stats import Gamma
from tests._polars_compat import assert_series_equal
from tests.scipy_parity._harness import Case, assert_case_matches_scipy

# `shape` spans the density regimes (diverging at `0` below `1`, the exponential at `1`, an interior
# mode above it) and both sides of the Stirling prefactor's switch at `shape = 10`; `rate` rescales the points.
_PARAMS = [(0.5, 2.0), (1.0, 1.0), (2.0, 1.5), (7.5, 0.5), (40.0, 4.0)]
# Endpoints excluded: `ppf(0) = 0` and `ppf(1) = +inf` are asserted in `inverse_test.py`.
_QUANTILES = [1e-10, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1 - 1e-10]

# Every method holds the default `1e-12` except `pdf`, the `exp` of the log density, whose relative error is the
# log's absolute error: `2.0e-15` relative (`2.3e-12` absolute) where the density reaches `1273` at `shape = 0.5`,
# against scipy's `2.4e-16`, both measured on an mpmath oracle.
_EXP_OF_LOG_DENSITY_TOL = 1e-11
_CASES: list[Case[Gamma]] = [
    Case("pdf", "value", lambda g, x: g.pdf(x), "pdf", tol=_EXP_OF_LOG_DENSITY_TOL),
    Case("log_pdf", "value", lambda g, x: g.log_pdf(x), "logpdf"),
    Case("cdf", "value", lambda g, x: g.cdf(x), "cdf"),
    Case("log_cdf", "value", lambda g, x: g.log_cdf(x), "logcdf"),
    Case("sf", "value", lambda g, x: g.sf(x), "sf"),
    Case("log_sf", "value", lambda g, x: g.log_sf(x), "logsf"),
    Case("ppf", "quantile", lambda g, x: g.ppf(x), "ppf"),
    Case("isf", "quantile", lambda g, x: g.isf(x), "isf"),
    Case("mean", "scalar", lambda g, _: g.mean(), "mean"),
    Case("variance", "scalar", lambda g, _: g.variance(), "var"),
    Case("std", "scalar", lambda g, _: g.std(), "std"),
    Case("median", "scalar", lambda g, _: g.median(), "median"),
    Case("entropy", "scalar", lambda g, _: g.entropy(), "entropy"),
]

_PROBABILITIES = (0.001, 0.05, 0.3, 0.5, 0.7, 0.95, 0.999)
"""The cdf levels the value grid sits at, so every method stays `O(10)` at every shape."""


def _value_grid(shape: float, rate: float) -> list[float]:
    """Below the support, then the points at `_PROBABILITIES` and one at `sf = e ** -20`.

    A grid in multiples of the mean would put `log_sf` far below `-1e3` at `shape = 40`, where the
    harness's absolute `1e-12` sits below one ulp. `0` itself is `support_test.py`'s.
    """
    frozen = scipy_gamma(a=shape, scale=1.0 / rate)
    return [-1.0 / rate, *(float(x) for x in frozen.ppf(_PROBABILITIES)), float(frozen.isf(2.0e-9))]


@pytest.mark.parametrize(("shape", "rate"), _PARAMS, ids=[f"shape={shape},rate={rate}" for shape, rate in _PARAMS])
@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
def test_method_matches_scipy(case: Case[Gamma], shape: float, rate: float) -> None:
    """Every method matches `scipy.stats.gamma(a=shape, scale=1 / rate)` across the parameter grid.

    Method-specific behaviour (null propagation, out-of-range quantiles, the support edge) is asserted
    in the shared contract files.
    """
    assert_case_matches_scipy(
        case,
        dist=Gamma(shape=shape, rate=rate),
        scipy_frozen=scipy_gamma(a=shape, scale=1.0 / rate),
        value_grid=_value_grid(shape, rate),
        quantiles=_QUANTILES,
    )


# Small `q`, where a fixed-budget Newton iteration on the cdf stalls on the vanishing density, and `q` near `1`,
# where only `1 - q` carries the quantile. scipy's `gammaincinv` holds `1.2e-14` relative across this sweep and
# this library `1.6e-14`, both on an mpmath oracle.
_INVERSE_SHAPES = [0.05, 0.5, 2.0, 100.0]
_INVERSE_QUANTILES = [1e-7, 1e-6, 1e-5, 0.9999, 1 - 1e-7]


@pytest.mark.parametrize("shape", _INVERSE_SHAPES, ids=lambda s: f"shape={s}")
@pytest.mark.parametrize("method", ["ppf", "isf"])
def test_inverses_match_scipy_in_both_tails(shape: float, method: Literal["ppf", "isf"]) -> None:
    """`ppf` and `isf` hold `1e-12` relative at every quantile of the sweep.

    Relative, not the harness's absolute tolerance: the quantiles span `1e-140` to `140`.
    """
    frame = pl.DataFrame({"q": _INVERSE_QUANTILES})
    got = frame.select(r=getattr(Gamma(shape=shape, rate=1.0), method)(pl.col("q")))["r"]
    expected = pl.Series(values=getattr(scipy_gamma(a=shape), method)(_INVERSE_QUANTILES))
    assert_series_equal(got, expected, rel_tol=1e-12, abs_tol=0.0, check_names=False)


# `(shape, q)` where statrs' `inverse_cdf` returns `8.7e-10`, `9.3e-10` and `9.3e-10` against true quantiles
# of `1.4e-15`, `7.9e-21` and `5.8e-101`.
_DEEP_LEFT_TAIL = [(2.0, 1e-30), (0.5, 1e-10), (0.05, 1e-5)]


@pytest.mark.parametrize(("shape", "q"), _DEEP_LEFT_TAIL, ids=str)
def test_ppf_resolves_a_quantile_far_below_the_bulk(shape: float, q: float) -> None:
    """The inverse iterates on `ln t`, so a quantile of `5.8e-101` keeps relative precision."""
    got = pl.select(r=Gamma(shape=shape, rate=1.0).ppf(q))["r"]
    expected = pl.Series(values=[scipy_gamma(a=shape).ppf(q)])
    assert_series_equal(got, expected, rel_tol=1e-12, abs_tol=0.0, check_names=False)
