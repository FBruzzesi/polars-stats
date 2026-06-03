from __future__ import annotations

from math import exp

import polars as pl
import pytest
from scipy.stats import lognorm as scipy_lognorm

from polars_stats import LogNormal
from tests.scipy_parity._harness import Case, assert_case_matches_scipy

# Parameter and evaluation grids for the parity sweep. Owned by this test category and independent
# of the per-method functional tests under `tests/distributions/lognormal`. `sigma` is kept moderate:
# the mean / variance grow exponentially in `sigma` and lose absolute precision against scipy past
# `sigma > 5` (see the issue caveat).
_PARAMS = [(0.0, 1.0), (0.5, 0.5), (-1.0, 0.25), (1.0, 0.75), (0.0, 0.1)]
_QUANTILES = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]

# Tolerance split, by upstream implementation rather than by aspiration:
#   * `statrs` evaluates pdf / ln_pdf to ~1e-15 against scipy, and the moments (exp-based) to
#     machine precision on this grid, so they hold the default 1e-12 target;
#   * cdf / sf (and their logs) and the closed-form ppf go through `erfc` / `erfc_inv`, while scipy
#     uses Cephes; the two agree only to ~1e-10. 1e-9 is the honest, comfortably-padded bound.
_TOL_ERF = 1e-9


def _scipy_frozen(mu: float, sigma: float) -> object:
    """`scipy.stats.lognorm` frozen at the polars-stats `(mu, sigma)`: `s=sigma`, `scale=exp(mu)`."""
    return scipy_lognorm(s=sigma, scale=exp(mu))


_CASES: list[Case[LogNormal]] = [
    Case("pdf", "value", lambda d, c: d.pdf(c), "pdf"),
    Case("log_pdf", "value", lambda d, c: d.log_pdf(c), "logpdf"),
    Case("cdf", "value", lambda d, c: d.cdf(c), "cdf", _TOL_ERF),
    Case("log_cdf", "value", lambda d, c: d.log_cdf(c), "logcdf", _TOL_ERF),
    Case("sf", "value", lambda d, c: d.sf(c), "sf", _TOL_ERF),
    Case("log_sf", "value", lambda d, c: d.log_sf(c), "logsf", _TOL_ERF),
    Case("ppf", "quantile", lambda d, c: d.ppf(c), "ppf", _TOL_ERF),
    Case("isf", "quantile", lambda d, c: d.isf(c), "isf", _TOL_ERF),
    Case("mean", "scalar", lambda d, _: d.mean(), "mean"),
    Case("variance", "scalar", lambda d, _: d.variance(), "var"),
    Case("std", "scalar", lambda d, _: d.std(), "std"),
    Case("median", "scalar", lambda d, _: d.median(), "median"),
    Case("entropy", "scalar", lambda d, _: d.entropy(), "entropy"),
]


def _value_grid(mu: float, sigma: float) -> list[float]:
    """Positive evaluation points, geometric around the median `exp(mu)`."""
    ks = pl.Series([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    return (mu + ks * sigma).exp().to_list()


@pytest.mark.parametrize(("mu", "sigma"), _PARAMS, ids=[f"mu={m},sigma={s}" for m, s in _PARAMS])
@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
def test_method_matches_scipy(case: Case[LogNormal], mu: float, sigma: float) -> None:
    """Every closed-form method matches `scipy.stats.lognorm(s=sigma, scale=exp(mu))` across the grid.

    Table-driven: adding a method is one row in `_CASES`. Method-specific behaviour (null
    propagation, out-of-range quantiles, the `x <= 0` support edge) is asserted in the per-method files.
    """
    assert_case_matches_scipy(
        case,
        dist=LogNormal(mu=mu, sigma=sigma),
        scipy_frozen=_scipy_frozen(mu, sigma),
        value_grid=_value_grid(mu, sigma),
        quantiles=_QUANTILES,
    )
