from __future__ import annotations

import polars as pl
import pytest
from scipy.stats import binom as scipy_binom

from polars_stats import Binomial
from tests.scipy_parity._harness import Case, assert_case_matches_scipy

# Parameter and evaluation grids for the parity sweep. Owned by this test category and independent
# of the per-method functional tests under `tests/distributions/binomial`.
_NS = [1, 5, 10]
_PROBS = [0.0, 0.25, 0.5, 0.75, 1.0]
# Interior quantiles only: scipy's discrete ppf/isf return the below-support sentinel -1 at the
# exact endpoints q in {0, 1}, which this support-clamped ppf does not reproduce.
_QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
# Spans below the support, several integer points, a non-integer, the upper bound, and above it.
_VALUE_GRID = [-1.0, 0.0, 1.0, 2.5, 3.0, 5.0, 10.0, 11.0]


_CASES: list[Case[Binomial]] = [
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
@pytest.mark.parametrize("n", _NS, ids=[f"n={n}" for n in _NS])
@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
def test_method_matches_scipy(case: Case[Binomial], n: int, p: float) -> None:
    """Every closed-form method matches `scipy.stats.binom(n, p)` across `_NS` x `_PROBS`.

    Table-driven: adding a method is one row in `_CASES`. `ppf`/`isf` use interior quantiles only,
    since scipy's discrete inverse returns the below-support sentinel -1 at `q` in `{0, 1}`, which the
    support-clamped `ppf` here does not reproduce. Method-specific behaviour (support handling, null
    propagation) is asserted in the per-method test files.
    """
    assert_case_matches_scipy(
        case,
        dist=Binomial(n, p),
        scipy_frozen=scipy_binom(n, p),
        value_grid=_VALUE_GRID,
        quantiles=_QUANTILES,
    )


# Regressions from `make audit`, in a regime `_QUANTILES` / `_VALUE_GRID` above never reach.


@pytest.mark.parametrize(("n", "p"), [(1000, 0.5), (5000, 0.001), (50, 1e-8), (1000, 0.999), (10, 0.5)])
@pytest.mark.parametrize("q", [1e-16, 1e-13, 1e-12, 1e-11, 1e-8, 0.5])
def test_ppf_resolves_quantiles_below_the_step_tolerance(n: int, p: float, q: float) -> None:
    """`ppf` matches scipy for a quantile far below the cdf step tolerance.

    The step slack absorbing the cdf's last-ULP error was *absolute* (`1e-12`), so it exceeded every
    quantile below it and `cdf(0) + tol >= q` held immediately: the search collapsed to `0` for all
    `q <= 1e-12` (`Binomial(1000, 0.5).ppf(1e-16)` returned `0` against scipy's `371`). Making the
    slack relative to `q` keeps its purpose and restores the whole lower tail.
    """
    got = pl.DataFrame({"q": [q]}).select(r=Binomial(n, p).ppf(pl.col("q")))["r"].item()
    assert got == float(scipy_binom.ppf(q, n, p))


@pytest.mark.parametrize(("n", "p"), [(1000, 0.5), (5000, 0.001), (50, 1e-8), (1000, 0.999), (10, 0.0), (10, 1.0)])
def test_ppf_at_one_returns_the_support_maximum(n: int, p: float) -> None:
    """`ppf(1)` is `n`, the documented support bound.

    `cdf(k)` reaches `1.0` well below `n` once the upper tail underflows, so a search stopped at the
    first such `k` (`Binomial(5000, 0.001).ppf(1.0)` returned `27`). scipy agrees on `n` here; only
    `q = 0`, where scipy returns its below-support sentinel `-1`, is deliberately not reproduced.
    """
    got = pl.DataFrame({"q": [1.0]}).select(r=Binomial(n, p).ppf(pl.col("q")))["r"].item()
    assert got == float(n) == float(scipy_binom.ppf(1.0, n, p))
