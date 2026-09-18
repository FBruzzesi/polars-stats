from __future__ import annotations

import pytest
from scipy.stats import cauchy as scipy_cauchy

from polars_stats import Cauchy
from tests.scipy_parity._harness import Case, assert_case_matches_scipy

# Parameter and evaluation grids for the parity sweep. Owned by this test category and independent.
_PARAMS = [(0.0, 1.0), (1.5, 2.0), (-3.0, 0.5), (100.0, 1e-3)]
# Endpoints excluded: `ppf(0) = -inf` and `ppf(1) = +inf` are asserted in `inverse_test.py`.
_QUANTILES = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]

# Every method is an elementary closed form (`arctan`, `tan`, `log`), so it holds the harness default
# `1e-12`. `mean`, `variance` and `std` are absent on purpose: they are null here where scipy returns
# `nan`, a contract `tests/distributions/moments_test.py` pins through `UNDEFINED_MOMENTS`.
_CASES: list[Case[Cauchy]] = [
    Case("pdf", "value", lambda c, x: c.pdf(x), "pdf"),
    Case("log_pdf", "value", lambda c, x: c.log_pdf(x), "logpdf"),
    Case("cdf", "value", lambda c, x: c.cdf(x), "cdf"),
    Case("log_cdf", "value", lambda c, x: c.log_cdf(x), "logcdf"),
    Case("sf", "value", lambda c, x: c.sf(x), "sf"),
    Case("log_sf", "value", lambda c, x: c.log_sf(x), "logsf"),
    Case("ppf", "quantile", lambda c, x: c.ppf(x), "ppf"),
    Case("isf", "quantile", lambda c, x: c.isf(x), "isf"),
    Case("median", "scalar", lambda c, _: c.median(), "median"),
    Case("entropy", "scalar", lambda c, _: c.entropy(), "entropy"),
]


def _value_grid(loc: float, scale: float) -> list[float]:
    """Evaluation points at `loc + k * scale`, both tails, stopping where scipy's naive `logsf` is still exact."""
    return [loc + k * scale for k in (-100.0, -8.0, -1.0, -0.5, 0.0, 0.5, 1.0, 8.0, 100.0)]


@pytest.mark.parametrize(("loc", "scale"), _PARAMS, ids=[f"loc={loc},scale={scale}" for loc, scale in _PARAMS])
@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
def test_method_matches_scipy(case: Case[Cauchy], loc: float, scale: float) -> None:
    """Every closed-form method matches `scipy.stats.cauchy(loc, scale)` across the parameter grid.

    Method-specific behaviour (null propagation, out-of-range quantiles, the infinite `ppf` endpoints)
    is asserted in the shared contract files.
    """
    assert_case_matches_scipy(
        case,
        dist=Cauchy(loc=loc, scale=scale),
        scipy_frozen=scipy_cauchy(loc=loc, scale=scale),
        value_grid=_value_grid(loc, scale),
        quantiles=_QUANTILES,
    )
