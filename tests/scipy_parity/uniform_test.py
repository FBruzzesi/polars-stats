from __future__ import annotations

import pytest
from scipy.stats import uniform as scipy_uniform

from polars_stats import Uniform
from tests.scipy_parity._harness import Case, assert_case_matches_scipy

# Parameter and evaluation grids for the parity sweep. Owned by this test category and independent
# of the per-method functional tests under `tests/distributions/uniform`.
_PARAMS = [(0.0, 1.0), (-2.0, 3.0), (2.0, 5.0), (-5.0, -1.0), (0.0, 1e-3)]
_QUANTILES = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]


_CASES: list[Case[Uniform]] = [
    Case("pdf", "value", lambda u, c: u.pdf(c), "pdf"),
    Case("log_pdf", "value", lambda u, c: u.log_pdf(c), "logpdf"),
    Case("cdf", "value", lambda u, c: u.cdf(c), "cdf"),
    Case("log_cdf", "value", lambda u, c: u.log_cdf(c), "logcdf"),
    Case("sf", "value", lambda u, c: u.sf(c), "sf"),
    Case("log_sf", "value", lambda u, c: u.log_sf(c), "logsf"),
    Case("ppf", "quantile", lambda u, c: u.ppf(c), "ppf"),
    Case("isf", "quantile", lambda u, c: u.isf(c), "isf"),
    Case("mean", "scalar", lambda u, _: u.mean(), "mean"),
    Case("variance", "scalar", lambda u, _: u.variance(), "var"),
    Case("std", "scalar", lambda u, _: u.std(), "std"),
    Case("median", "scalar", lambda u, _: u.median(), "median"),
    Case("entropy", "scalar", lambda u, _: u.entropy(), "entropy"),
]


def _value_grid(mn: float, mx: float) -> list[float]:
    """Evaluation points for a `(min, max)` support: below, both endpoints, interior, above."""
    width = mx - mn
    return [mn - width, mn, mn + 0.25 * width, (mn + mx) / 2, mn + 0.75 * width, mx, mx + width]


@pytest.mark.parametrize(("mn", "mx"), _PARAMS, ids=[f"min={mn},max={mx}" for mn, mx in _PARAMS])
@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
def test_method_matches_scipy(case: Case[Uniform], mn: float, mx: float) -> None:
    """Every closed-form method matches `scipy.stats.uniform` across the `(min, max)` grid.

    Table-driven: adding a method is one row in `_CASES`. Method-specific behaviour (support
    clamping, null propagation, out-of-range quantiles) is asserted in the per-method test files.
    """
    assert_case_matches_scipy(
        case,
        dist=Uniform(min=mn, max=mx),
        scipy_frozen=scipy_uniform(loc=mn, scale=mx - mn),
        value_grid=_value_grid(mn, mx),
        quantiles=_QUANTILES,
    )
