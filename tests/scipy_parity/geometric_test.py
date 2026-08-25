from __future__ import annotations

import math

import polars as pl
import pytest
from scipy.stats import geom as scipy_geom

from polars_stats import Geometric
from tests._polars_compat import assert_series_equal
from tests.scipy_parity._harness import Case, assert_case_matches_scipy

# Parameter and evaluation grids for the parity sweep. Owned by this test category and independent
# of the per-method functional tests under `tests/distributions/geometric`.
# The support is `0 < p <= 1`; scipy accepts `p = 0` (degenerate answers) but this crate rejects it,
# so the grid holds only valid parameters. `p = 1` is excluded because scipy's generic discrete
# entropy does not answer there (see the divergence note below).
_PS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9]
# Interior quantiles only: scipy's discrete ppf/isf return the below-support sentinel -1 at the
# exact endpoints q in {0, 1}, which this support-clamped ppf/isf does not reproduce.
_QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
# Spans below the support, several integer points, a non-integer, and well into the upper tail.
_VALUE_GRID = [-1.0, 0.0, 1.0, 2.0, 3.7, 10.0, 50.0]


_CASES: list[Case[Geometric]] = [
    Case("pmf", "value", lambda g, c: g.pmf(c), "pmf"),
    Case("log_pmf", "value", lambda g, c: g.log_pmf(c), "logpmf"),
    Case("cdf", "value", lambda g, c: g.cdf(c), "cdf"),
    Case("log_cdf", "value", lambda g, c: g.log_cdf(c), "logcdf"),
    Case("sf", "value", lambda g, c: g.sf(c), "sf"),
    Case("log_sf", "value", lambda g, c: g.log_sf(c), "logsf"),
    Case("ppf", "quantile", lambda g, c: g.ppf(c), "ppf"),
    Case("isf", "quantile", lambda g, c: g.isf(c), "isf"),
    Case("mean", "scalar", lambda g, _: g.mean(), "mean"),
    Case("variance", "scalar", lambda g, _: g.variance(), "var"),
    Case("std", "scalar", lambda g, _: g.std(), "std"),
    Case("median", "scalar", lambda g, _: g.median(), "median"),
    Case("entropy", "scalar", lambda g, _: g.entropy(), "entropy"),
]


@pytest.mark.parametrize("p", _PS, ids=[f"p={p}" for p in _PS])
@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
def test_method_matches_scipy(case: Case[Geometric], p: float) -> None:
    """Every closed-form method matches `scipy.stats.geom(p)` across `_PS` x grids.

    Table-driven: adding a method is one row in `_CASES`. `ppf`/`isf` use interior quantiles only,
    since scipy's discrete inverse returns the below-support sentinel -1 at `q` in `{0, 1}`, which
    the support-clamped inverses here do not reproduce. Method-specific behaviour (support handling,
    null propagation) is asserted in the per-method test files.
    """
    assert_case_matches_scipy(
        case,
        dist=Geometric(p),
        scipy_frozen=scipy_geom(p),
        value_grid=_VALUE_GRID,
        quantiles=_QUANTILES,
    )


# Regressions from `make audit`, in a regime `_QUANTILES` / `_VALUE_GRID` above never reach.


def test_log_sf_deep_tail_beyond_the_linear_underflow() -> None:
    """`log_sf` keeps digits where `(1 - p)**k` underflows to exactly `0.0`.

    At `p = 0.001`, the linear sf hits `0.0` around `k ~ 70_000`; a naive `log(sf)` would answer
    `-inf` from there on. The closed form stays exact, so this compares against it directly rather
    than against scipy, which is not a trustworthy oracle past its own saturation.
    """
    p = 0.001
    for k in [100_000, 1_000_000]:
        got = pl.DataFrame({"k": [float(k)]}).select(r=Geometric(p).log_sf(pl.col("k")))["r"].item()
        assert got == pytest.approx(k * math.log1p(-p))


def test_ppf_resolves_quantiles_far_below_one_half() -> None:
    """`ppf` matches scipy for quantiles where a collapsed `log(1 - q)` would answer `1`."""
    for p, q in [(0.5, 1e-16), (0.001, 1e-13), (0.9, 1e-20)]:
        got = pl.DataFrame({"q": [q]}).select(r=Geometric(p).ppf(pl.col("q")))["r"].item()
        assert got == float(scipy_geom.ppf(q, p))


def test_isf_deep_tail_keeps_full_precision() -> None:
    """`isf` enters against `q` itself, so it resolves far past the `1 - q` quantisation.

    Asserted against the exact closed form only: scipy's discrete inverse goes through
    `ppf(1 - q)`, whose complement saturates here (`log1p(0)` divides by zero and scipy answers
    `inf`), so it is not a usable oracle in this regime — which is the defect this override exists
    to avoid.
    """
    p = 0.5
    q = 1e-300
    expected = math.ceil(math.log(q) / math.log1p(-p))
    got = pl.DataFrame({"q": [q]}).select(r=Geometric(p).isf(pl.col("q")))["r"].item()
    assert got == float(expected)


def test_memorylessness_of_sf() -> None:
    """`sf(k + j) = sf(k) * sf(j)`, the identity that names the distribution memoryless.

    Holds only up to float rounding of the shared product; asserted to the default tolerance.
    """
    p = 0.3
    df = pl.DataFrame({"k": [1.0, 5.0, 10.0], "j": [3.0, 7.0, 25.0]})
    sf_kj = df.select(r=Geometric(p).sf(pl.col("k") + pl.col("j")))["r"]
    sf_k_times_j = df.select(r=Geometric(p).sf(pl.col("k")) * Geometric(p).sf(pl.col("j")))["r"]
    assert_series_equal(sf_kj, sf_k_times_j, check_names=False)


# NOTE: Deliberate divergences from scipy, recorded here rather than rediscovered as surprises.
#
# * Endpoints of the inverse: scipy's discrete `ppf(0)` returns its below-support sentinel `-1`,
#   while the smallest support point here (`k = 1`) is the only meaningful answer; `ppf(1)` is `inf`
#   on both sides. Interior-quantile parity covers everything else.
# * `p = 0`: scipy accepts it and returns degenerate values; this crate raises, as the geometric
#   law degenerates to "never succeeds" and has no finite moments.
# * `entropy` at `p = 1`: this crate answers `0.0` by the `0 * log 0 = 0` convention; scipy's
#   generic discrete entropy does not answer there.
# * Deep-tail `isf`: scipy quantises the tail mass through `ppf(1 - q)` and saturates (`inf`) for
#   `q` below roughly 1e-16; this crate inverts against `q` directly and stays finite (asserted
#   above against the exact closed form, since scipy cannot referee there).
