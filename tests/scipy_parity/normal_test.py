from __future__ import annotations

import pytest
from scipy.stats import norm as scipy_norm

from polars_stats import Normal
from tests.scipy_parity._harness import Case, assert_case_matches_scipy

# Parameter and evaluation grids for the parity sweep. Owned by this test category and independent
# of the per-method functional tests under `tests/distributions/normal`.
_PARAMS = [(0.0, 1.0), (1.0, 2.0), (-3.0, 0.5), (10.0, 5.0), (0.0, 1e-3)]
_QUANTILES = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]

# Tolerance split, by upstream implementation rather than by aspiration:
#   * `statrs` evaluates pdf / ln_pdf / inverse_cdf and the moments to ~1e-13 against scipy,
#     so they hold the default 1e-12 target;
#   * cdf / sf (and their logs) go through `erfc`, while scipy uses the Cephes `ndtr`; the two
#     agree only to ~1e-10. 1e-9 is the honest, comfortably-padded bound for that family.
_TOL_ERF = 1e-9


_CASES: list[Case[Normal]] = [
    Case("pdf", "value", lambda n, c: n.pdf(c), "pdf"),
    Case("log_pdf", "value", lambda n, c: n.log_pdf(c), "logpdf"),
    Case("cdf", "value", lambda n, c: n.cdf(c), "cdf", _TOL_ERF),
    Case("log_cdf", "value", lambda n, c: n.log_cdf(c), "logcdf", _TOL_ERF),
    Case("sf", "value", lambda n, c: n.sf(c), "sf", _TOL_ERF),
    Case("log_sf", "value", lambda n, c: n.log_sf(c), "logsf", _TOL_ERF),
    Case("ppf", "quantile", lambda n, c: n.ppf(c), "ppf"),
    Case("isf", "quantile", lambda n, c: n.isf(c), "isf"),
    Case("mean", "scalar", lambda n, _: n.mean(), "mean"),
    Case("variance", "scalar", lambda n, _: n.variance(), "var"),
    Case("std", "scalar", lambda n, _: n.std(), "std"),
    Case("median", "scalar", lambda n, _: n.median(), "median"),
    Case("entropy", "scalar", lambda n, _: n.entropy(), "entropy"),
]


def _value_grid(mean: float, std: float) -> list[float]:
    """Evaluation points spanning both tails through the centre of a `(mean, std)` distribution."""
    return [mean - 3 * std, mean - std, mean - 0.25 * std, mean, mean + 0.25 * std, mean + std, mean + 3 * std]


@pytest.mark.parametrize(("mean", "std"), _PARAMS, ids=[f"mean={m},std={s}" for m, s in _PARAMS])
@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
def test_method_matches_scipy(case: Case[Normal], mean: float, std: float) -> None:
    """Every closed-form method matches `scipy.stats.norm` across the `(mean, std)` grid.

    Table-driven: adding a method is one row in `_CASES`. Method-specific behaviour (null
    propagation, out-of-range quantiles, infinite ppf endpoints) is asserted in the per-method files.
    """
    assert_case_matches_scipy(
        case,
        dist=Normal(mean=mean, std_dev=std),
        scipy_frozen=scipy_norm(loc=mean, scale=std),
        value_grid=_value_grid(mean, std),
        quantiles=_QUANTILES,
    )
