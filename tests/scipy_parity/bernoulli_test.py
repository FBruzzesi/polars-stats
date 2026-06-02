from __future__ import annotations

import pytest
from scipy.stats import bernoulli as scipy_bernoulli

from polars_stats import Bernoulli
from tests.scipy_parity._harness import Case, assert_case_matches_scipy

# Parameter and evaluation grids for the parity sweep. Owned by this test category and independent
# of the per-method functional tests under `tests/distributions/bernoulli`.
_PROBS = [0.0, 0.25, 0.5, 0.75, 1.0]
# Interior quantiles only: scipy's discrete ppf/isf return the below-support sentinel -1 at the
# exact endpoints q in {0, 1}, which the Boolean-valued `ppf` here (range {False, True}) does not
# reproduce. The interior is where the two definitions agree.
_QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
# Bernoulli support is {0, 1}; the grid spans below, both support points, a non-integer, and above.
_VALUE_GRID = [-1.0, 0.0, 0.5, 1.0, 2.0]


_CASES: list[Case[Bernoulli]] = [
    Case("pmf", "value", lambda b, c: b.pmf(c), "pmf"),
    Case("log_pmf", "value", lambda b, c: b.log_pmf(c), "logpmf"),
    Case("cdf", "value", lambda b, c: b.cdf(c), "cdf"),
    Case("log_cdf", "value", lambda b, c: b.log_cdf(c), "logcdf"),
    Case("sf", "value", lambda b, c: b.sf(c), "sf"),
    Case("log_sf", "value", lambda b, c: b.log_sf(c), "logsf"),
    Case("ppf", "quantile", lambda b, c: b.ppf(c), "ppf"),
    Case("isf", "quantile", lambda b, c: b.isf(c), "isf"),
    Case("mean", "scalar", lambda b, _: b.mean(), "mean"),
    Case("variance", "scalar", lambda b, _: b.variance(), "var"),
    Case("std", "scalar", lambda b, _: b.std(), "std"),
    Case("median", "scalar", lambda b, _: b.median(), "median"),
    Case("entropy", "scalar", lambda b, _: b.entropy(), "entropy"),
]


@pytest.mark.parametrize("p", _PROBS, ids=[f"p={p}" for p in _PROBS])
@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
def test_method_matches_scipy(case: Case[Bernoulli], p: float) -> None:
    """Every closed-form method matches `scipy.stats.bernoulli` across `_PROBS`.

    Table-driven: adding a method is one row in `_CASES`. `ppf`/`isf` use interior quantiles only,
    since scipy's discrete inverse returns the below-support sentinel -1 at `q` in `{0, 1}`, which
    the Boolean-valued `ppf` here does not reproduce. Method-specific behaviour (support handling,
    null propagation) is asserted in the per-method test files.
    """
    assert_case_matches_scipy(
        case,
        dist=Bernoulli(p=p),
        scipy_frozen=scipy_bernoulli(p),
        value_grid=_VALUE_GRID,
        quantiles=_QUANTILES,
    )
