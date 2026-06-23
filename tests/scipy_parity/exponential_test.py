from __future__ import annotations

import pytest
from scipy.stats import expon as scipy_expon

from polars_stats import Exponential
from tests.scipy_parity._harness import Case, assert_case_matches_scipy

# Parameter and evaluation grids for the parity sweep. Owned by this test category and independent
# of the per-method functional tests under `tests/distributions/exponential`.
_PARAMS = [0.1, 0.5, 1.0, 2.0, 5.0]
# Endpoints excluded: `ppf(1) = +inf` (the unbounded right tail) is asserted in `ppf_test.py`.
_QUANTILES = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]


# Exponential is elementary closed-form (no special functions), so every method agrees with scipy to
# machine precision and holds the harness default 1e-12. The `value_grid` stays on the open support
# `x > 0`: at `x <= 0` (`cdf = 0`) `log_cdf` is `-inf`, asserted in `log_cdf_test.py` instead.
_CASES: list[Case[Exponential]] = [
    Case("pdf", "value", lambda e, c: e.pdf(c), "pdf"),
    Case("log_pdf", "value", lambda e, c: e.log_pdf(c), "logpdf"),
    Case("cdf", "value", lambda e, c: e.cdf(c), "cdf"),
    Case("log_cdf", "value", lambda e, c: e.log_cdf(c), "logcdf"),
    Case("sf", "value", lambda e, c: e.sf(c), "sf"),
    Case("log_sf", "value", lambda e, c: e.log_sf(c), "logsf"),
    Case("ppf", "quantile", lambda e, c: e.ppf(c), "ppf"),
    Case("isf", "quantile", lambda e, c: e.isf(c), "isf"),
    Case("mean", "scalar", lambda e, _: e.mean(), "mean"),
    Case("variance", "scalar", lambda e, _: e.variance(), "var"),
    Case("std", "scalar", lambda e, _: e.std(), "std"),
    Case("median", "scalar", lambda e, _: e.median(), "median"),
    Case("entropy", "scalar", lambda e, _: e.entropy(), "entropy"),
]


def _value_grid(rate: float) -> list[float]:
    """Positive evaluation points, in multiples of the mean `1 / rate` (so `rate * x` is grid-fixed)."""
    mean = 1.0 / rate
    return [0.1 * mean, 0.25 * mean, 0.5 * mean, mean, 2.0 * mean, 4.0 * mean, 8.0 * mean]


@pytest.mark.parametrize("rate", _PARAMS, ids=[f"rate={r}" for r in _PARAMS])
@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
def test_method_matches_scipy(case: Case[Exponential], rate: float) -> None:
    """Every closed-form method matches `scipy.stats.expon(scale=1 / rate)` across the rate grid.

    Table-driven: adding a method is one row in `_CASES`. Method-specific behaviour (support edge,
    null propagation, out-of-range quantiles, the infinite ppf endpoint) is asserted in the
    per-method test files.
    """
    assert_case_matches_scipy(
        case,
        dist=Exponential(rate=rate),
        scipy_frozen=scipy_expon(scale=1.0 / rate),
        value_grid=_value_grid(rate),
        quantiles=_QUANTILES,
    )
