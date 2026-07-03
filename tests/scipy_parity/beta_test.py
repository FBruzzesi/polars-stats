from __future__ import annotations

import pytest
from scipy.stats import beta as scipy_beta

from polars_stats import Beta
from tests.scipy_parity._harness import Case, assert_case_matches_scipy

# Parameter and evaluation grids for the parity sweep. Owned by this test category and independent
# of the per-method functional tests under `tests/distributions/beta`. The grid covers the distinct
# shape regimes: unimodal, U-shaped (both shapes < 1), uniform (1, 1), J-shaped, skewed, and the
# large-shape regime where statrs switches its pdf to `ln_pdf().exp()`.
_PARAMS = [(2.0, 3.0), (0.5, 0.5), (1.0, 1.0), (5.0, 1.0), (2.0, 8.0), (90.0, 100.0)]
_QUANTILES = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
# The support is fixed to [0, 1], so one interior grid spanning both tails serves every shape pair.
_VALUE_GRID = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]

# Tolerance split, by upstream implementation rather than by aspiration:
#   * cdf / sf (and their logs), ppf / isf (AS 64 + Newton inverse), the moments and the entropy
#     agree with scipy's Cephes incomplete-beta family to ~1e-13 across the grid, holding the
#     default 1e-12 target;
#   * pdf goes through `ln_pdf().exp()` in statrs for shapes > 80, which costs a few digits of
#     absolute accuracy at the large-shape pair (~1e-12 observed). 1e-11 is the honest, padded bound.
_TOL_LARGE_SHAPE_PDF = 1e-11


_CASES: list[Case[Beta]] = [
    Case("pdf", "value", lambda d, c: d.pdf(c), "pdf", _TOL_LARGE_SHAPE_PDF),
    Case("log_pdf", "value", lambda d, c: d.log_pdf(c), "logpdf"),
    Case("cdf", "value", lambda d, c: d.cdf(c), "cdf"),
    Case("log_cdf", "value", lambda d, c: d.log_cdf(c), "logcdf"),
    Case("sf", "value", lambda d, c: d.sf(c), "sf"),
    Case("log_sf", "value", lambda d, c: d.log_sf(c), "logsf"),
    Case("ppf", "quantile", lambda d, c: d.ppf(c), "ppf"),
    Case("isf", "quantile", lambda d, c: d.isf(c), "isf"),
    Case("mean", "scalar", lambda d, _: d.mean(), "mean"),
    Case("variance", "scalar", lambda d, _: d.variance(), "var"),
    Case("std", "scalar", lambda d, _: d.std(), "std"),
    Case("median", "scalar", lambda d, _: d.median(), "median"),
    Case("entropy", "scalar", lambda d, _: d.entropy(), "entropy"),
]


@pytest.mark.parametrize(("a", "b"), _PARAMS, ids=[f"a={a},b={b}" for a, b in _PARAMS])
@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
def test_method_matches_scipy(case: Case[Beta], a: float, b: float) -> None:
    """Every closed-form method matches `scipy.stats.beta` across the `(a, b)` grid.

    Table-driven: adding a method is one row in `_CASES`. Method-specific behaviour (null
    propagation, out-of-range quantiles, the support boundaries) is asserted in the per-method files.
    """
    assert_case_matches_scipy(
        case,
        dist=Beta(a=a, b=b),
        scipy_frozen=scipy_beta(a=a, b=b),
        value_grid=_VALUE_GRID,
        quantiles=_QUANTILES,
    )
