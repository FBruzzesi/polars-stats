from __future__ import annotations

import math

import polars as pl
import pytest
from scipy.stats import bernoulli as scipy_bernoulli

from polars_stats import Bernoulli
from tests.scipy_parity._harness import Case, assert_case_matches_scipy

# Parameter and evaluation grids for the parity sweep. Owned by this test category and independent
# of the per-method functional tests under `tests/distributions/bernoulli`.
_PROBS = [0.0, 0.25, 0.5, 0.75, 1.0]
# Interior quantiles only: scipy's discrete ppf/isf return the below-support sentinel -1 at the
# exact endpoints q in {0, 1}, which the support-clamped `ppf` here (range {0.0, 1.0}) does not
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
    the support-clamped `ppf` here does not reproduce. Method-specific behaviour (support handling,
    null propagation) is asserted in the per-method test files.
    """
    assert_case_matches_scipy(
        case,
        dist=Bernoulli(p=p),
        scipy_frozen=scipy_bernoulli(p),
        value_grid=_VALUE_GRID,
        quantiles=_QUANTILES,
    )


# Regressions from `make audit`: the whole small-`p` regime, which `_PROBS` above does not reach.
# Every oracle is the exact closed form, since scipy's own `sf` / `logcdf` / `entropy` make the same
# `1 - (1 - p)` and `log(1 - p)` mistakes this fixes.
_TINY_PROBS = [1e-300, 1e-100, 1e-16, 1e-12, 1e-6]


@pytest.mark.parametrize("p", _TINY_PROBS, ids=lambda p: f"p={p}")
def test_small_p_survives_every_method(p: float) -> None:
    """`sf`, `log_sf`, `log_cdf`, `log_pmf` and `entropy` all keep a `p` far below the `1 - p` floor.

    Every one composed through `1 - p` or `log(1 - p)`, both of which round to `1` and `0` below
    `p ~ 1.1e-16`: `sf(0)` was `1 - (1 - p)`, so `Bernoulli(1e-300).sf(0.5)` returned `0.0` where the
    answer is `1e-300`, and at `p = 1e-16` it was 11% wrong. `sf` is the anomaly-scoring path, so
    this is the regime that matters most.
    """
    dist = Bernoulli(p=p)
    got = pl.DataFrame({"x": [0.0]}).select(
        sf=dist.sf(pl.col("x")),
        log_sf=dist.log_sf(pl.col("x")),
        log_cdf=dist.log_cdf(pl.col("x")),
        log_pmf=dist.log_pmf(pl.col("x")),
        entropy=dist.entropy(),
    )
    assert got["sf"].item() == pytest.approx(p, rel=1e-15)
    assert got["log_sf"].item() == pytest.approx(math.log(p), rel=1e-15)
    assert got["log_cdf"].item() == pytest.approx(math.log1p(-p), rel=1e-14)
    assert got["log_pmf"].item() == pytest.approx(math.log1p(-p), rel=1e-14)
    assert got["entropy"].item() == pytest.approx(-p * math.log(p) - (1 - p) * math.log1p(-p), rel=1e-13)


# The `ppf(1)` / `isf` regression, in the one parameter gap a 300-decade sweep still left open.
# Both went through a complement that saturates: `ppf` compared `quantile > 1 - p`, and `isf` was the
# base-class `ppf(1 - quantile)`, so it saturated twice. Below `p ~ 1.1e-16` the expression `1 - p` is
# exactly `1.0`, the comparison is false, and the answer is the wrong support point.
#
# `p = 1e-17` is the case that shows it: small enough that `1 - p` rounds to `1.0` (which `1e-16` does
# not), large enough that a quantile can be smaller still (which `1e-300` does not). Oracled by the
# definition (`isf(q) = 1` iff `sf(0) = p > q`), which needs no arithmetic at all.
_SATURATING_P = [1e-17, 1e-30, 1e-300]


@pytest.mark.parametrize("p", _SATURATING_P, ids=str)
@pytest.mark.parametrize("q", [0.0, 1e-300, 1e-40, 1e-20, 1e-18])
def test_isf_resolves_a_probability_below_the_complement_resolution(p: float, q: float) -> None:
    """`isf(q)` is `1` exactly when `p > q`, with no complement formed anywhere."""
    got = pl.DataFrame({"q": [q]}).select(r=Bernoulli(p=p).isf(pl.col("q")))["r"].item()
    assert got == (0.0 if p <= q else 1.0)


@pytest.mark.parametrize("p", _SATURATING_P, ids=str)
def test_ppf_at_one_returns_the_upper_support_point_for_a_tiny_p(p: float) -> None:
    """`ppf(1)` is the largest support point. It answered `0.0`, where scipy answers `1.0`."""
    got = pl.DataFrame({"q": [1.0]}).select(r=Bernoulli(p=p).ppf(pl.col("q")))["r"].item()
    assert got == 1.0
    assert got == float(scipy_bernoulli.ppf(1.0, p))


def test_ppf_at_one_stays_zero_for_the_degenerate_p() -> None:
    """`Bernoulli(0)` has one support point, so `ppf(1)` is `0`. scipy returns `1.0` here and is wrong."""
    got = pl.DataFrame({"q": [1.0]}).select(r=Bernoulli(p=0.0).ppf(pl.col("q")))["r"].item()
    assert got == 0.0
