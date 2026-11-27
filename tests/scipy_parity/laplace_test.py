from __future__ import annotations

import pytest
from scipy.stats import laplace as scipy_laplace

from polars_stats import Laplace
from tests.scipy_parity._harness import Case, assert_case_matches_scipy

# Parameter and evaluation grids for the parity sweep. Owned by this test category and independent.
_PARAMS = [(0.0, 1.0), (1.5, 2.0), (-3.0, 0.5), (100.0, 1e-3)]
# Endpoints excluded: `ppf(0) = -inf` and `ppf(1) = +inf` are asserted in `inverse_test.py`.
_QUANTILES = [1e-12, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1 - 1e-12]

# Every method is an elementary closed form (`exp`, `log`, `log1p`), so it holds the harness default `1e-12`.
_CASES: list[Case[Laplace]] = [
    Case("pdf", "value", lambda d, x: d.pdf(x), "pdf"),
    Case("log_pdf", "value", lambda d, x: d.log_pdf(x), "logpdf"),
    Case("cdf", "value", lambda d, x: d.cdf(x), "cdf"),
    Case("log_cdf", "value", lambda d, x: d.log_cdf(x), "logcdf"),
    Case("sf", "value", lambda d, x: d.sf(x), "sf"),
    Case("log_sf", "value", lambda d, x: d.log_sf(x), "logsf"),
    Case("ppf", "quantile", lambda d, x: d.ppf(x), "ppf"),
    Case("isf", "quantile", lambda d, x: d.isf(x), "isf"),
    Case("mean", "scalar", lambda d, _: d.mean(), "mean"),
    Case("variance", "scalar", lambda d, _: d.variance(), "var"),
    Case("std", "scalar", lambda d, _: d.std(), "std"),
    Case("median", "scalar", lambda d, _: d.median(), "median"),
    Case("entropy", "scalar", lambda d, _: d.entropy(), "entropy"),
]


def _value_grid(loc: float, scale: float) -> list[float]:
    """Evaluation points at `loc + k * scale`, both tails, stopping where scipy's `logpdf` and `logsf` still answer.

    scipy logs its density and survival mass rather than reading the exponent, so both are `-inf` once the mass
    underflows, past `|k| ~ 745`.
    """
    return [loc + k * scale for k in (-700.0, -100.0, -8.0, -1.0, -0.5, 0.0, 0.5, 1.0, 8.0, 100.0, 700.0)]


@pytest.mark.parametrize(("loc", "scale"), _PARAMS, ids=[f"loc={loc},scale={scale}" for loc, scale in _PARAMS])
@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
def test_method_matches_scipy(case: Case[Laplace], loc: float, scale: float) -> None:
    """Every method matches `scipy.stats.laplace(loc, scale)` across the parameter grid.

    Method-specific behaviour (null propagation, out-of-range quantiles, the infinite `ppf` endpoints)
    is asserted in the shared contract files.
    """
    assert_case_matches_scipy(
        case,
        dist=Laplace(loc=loc, scale=scale),
        scipy_frozen=scipy_laplace(loc=loc, scale=scale),
        value_grid=_value_grid(loc, scale),
        quantiles=_QUANTILES,
    )
