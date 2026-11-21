from __future__ import annotations

import math

import polars as pl
import pytest
from scipy.stats import nbinom as scipy_nbinom

from polars_stats import NegativeBinomial
from tests.scipy_parity._harness import Case, assert_case_matches_scipy

# scipy's `n` is `r` here. A non-integer `r` is the Polya extension, which scipy evaluates too.
_RS = [0.5, 1.0, 2.5, 10.0, 300.0]
_PS = [0.01, 0.3, 0.9]
# Interior quantiles only: scipy's discrete inverses answer the below-support sentinel `-1` at `q = 0`.
_QUANTILES = [0.001, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.999]

# scipy's `logpmf` is a `gammaln` difference: against mpmath it is `7.3e-11` off at `r = 300, p = 0.01`, where
# this one is `1.8e-15`. The tails and `ppf` / `isf` are the incomplete beta in both and agree to ~1e-15.
_TOL_LOG = 1e-10

# Below this `ln pmf(0) = r ln p` scipy's linear `cdf` near `0` underflows and its naive `logcdf` is `-inf`.
_LN_REPRESENTABLE = -700.0


def _value_grid(r: float, p: float) -> list[float]:
    """Below the support, a non-integer, and points out to eight standard deviations past the mean.

    The points next to `0` are left out where `pmf(0)` underflows; the audit owns that regime.
    """
    mean = r * (1 - p) / p
    sd = math.sqrt(r * (1 - p)) / p
    bulk = {float(max(0, math.floor(mean + k * sd))) for k in (-2, -1, 0, 1, 2, 4, 8)}
    near_zero = [0.0, 1.0, 2.5, 3.0] if r * math.log(p) > _LN_REPRESENTABLE else []
    return sorted({-1.0, *near_zero, *bulk})


_CASES: list[Case[NegativeBinomial]] = [
    Case("pmf", "value", lambda d, c: d.pmf(c), "pmf"),
    Case("log_pmf", "value", lambda d, c: d.log_pmf(c), "logpmf", _TOL_LOG),
    Case("cdf", "value", lambda d, c: d.cdf(c), "cdf"),
    Case("log_cdf", "value", lambda d, c: d.log_cdf(c), "logcdf", _TOL_LOG),
    Case("sf", "value", lambda d, c: d.sf(c), "sf"),
    Case("log_sf", "value", lambda d, c: d.log_sf(c), "logsf", _TOL_LOG),
    Case("ppf", "quantile", lambda d, c: d.ppf(c), "ppf", 0.0),
    Case("isf", "quantile", lambda d, c: d.isf(c), "isf", 0.0),
    Case("mean", "scalar", lambda d, _: d.mean(), "mean"),
    Case("variance", "scalar", lambda d, _: d.variance(), "var"),
    Case("std", "scalar", lambda d, _: d.std(), "std"),
    Case("median", "scalar", lambda d, _: d.median(), "median", 0.0),
]


@pytest.mark.parametrize("p", _PS, ids=[f"p={p}" for p in _PS])
@pytest.mark.parametrize("r", _RS, ids=[f"r={r}" for r in _RS])
@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
def test_method_matches_scipy(case: Case[NegativeBinomial], r: float, p: float) -> None:
    """Every method but `entropy` matches `scipy.stats.nbinom(n=r, p=p)`; `entropy` raises, see the registry."""
    assert_case_matches_scipy(
        case,
        dist=NegativeBinomial(r=r, p=p),
        scipy_frozen=scipy_nbinom(n=r, p=p),
        value_grid=_value_grid(r, p),
        quantiles=_QUANTILES,
    )


@pytest.mark.parametrize(("r", "p"), [(2.5, 0.3), (300.0, 0.01), (0.5, 0.9)])
def test_log_tails_stay_finite_beyond_the_linear_underflow(r: float, p: float) -> None:
    """Far past the mean `sf` underflows and scipy's `logsf`, the naive log, is `-inf`; `log_sf` keeps going.

    `identities_test.py` pins the value to `Beta(r, k + 1).log_sf(p)` bit for bit and the audit to mpmath; here it is
    enough that it is finite and decreasing past where scipy has given up.
    """
    mean = r * (1 - p) / p
    ks = [math.floor(mean + depth / -math.log1p(-p)) for depth in (100.0, 1000.0, 5000.0)]
    got = pl.DataFrame({"k": ks}).select(r=NegativeBinomial(r=r, p=p).log_sf(pl.col("k")))["r"].to_list()
    assert all(math.isfinite(value) for value in got)
    assert got == sorted(got, reverse=True)
    assert math.isinf(float(scipy_nbinom.logsf(ks[-1], r, p)))


@pytest.mark.parametrize("method", ["ppf", "isf"])
def test_an_inverse_whose_tail_runs_out_of_terms_is_nan(method: str) -> None:
    """At `r = 1e17, p = 1/2` the median sits where the incomplete beta runs out of terms: `NaN`, not a count."""
    got = pl.DataFrame({"q": [0.5]}).select(r=getattr(NegativeBinomial(r=1e17, p=0.5), method)(pl.col("q")))["r"]
    assert math.isnan(got.item())
