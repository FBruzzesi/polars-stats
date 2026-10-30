from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
from scipy.special import logsumexp
from scipy.stats import poisson as scipy_poisson

from polars_stats import Poisson
from tests._polars_compat import assert_series_equal
from tests._registry import poisson_support_limit
from tests.scipy_parity._harness import Case, assert_case_matches_scipy

# `rate` spans the mass piling onto `0` (`1e-3`), the unit rate, an interior mode, and both sides of the incomplete
# gamma's Temme switch at shape `100`. It stops at `200`: scipy's `logpmf` is `-rate + k ln(rate) - ln(k!)`, which
# cancels to `1.1e-12` absolute at `rate = 1e3`, so a larger `rate` would test scipy rather than this library.
_RATES = [1e-3, 0.5, 1.0, 4.5, 30.0, 200.0]
# Interior quantiles only: scipy's discrete `ppf(0)` is the `-1` sentinel below the support.
_QUANTILES = [1e-10, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1 - 1e-10]

_CASES: list[Case[Poisson]] = [
    Case("pmf", "value", lambda d, k: d.pmf(k), "pmf"),
    Case("log_pmf", "value", lambda d, k: d.log_pmf(k), "logpmf"),
    Case("cdf", "value", lambda d, k: d.cdf(k), "cdf"),
    Case("log_cdf", "value", lambda d, k: d.log_cdf(k), "logcdf"),
    Case("sf", "value", lambda d, k: d.sf(k), "sf"),
    Case("log_sf", "value", lambda d, k: d.log_sf(k), "logsf"),
    Case("ppf", "quantile", lambda d, q: d.ppf(q), "ppf"),
    Case("isf", "quantile", lambda d, q: d.isf(q), "isf"),
    Case("mean", "scalar", lambda d, _: d.mean(), "mean"),
    Case("variance", "scalar", lambda d, _: d.variance(), "var"),
    Case("std", "scalar", lambda d, _: d.std(), "std"),
    Case("median", "scalar", lambda d, _: d.median(), "median"),
    Case("entropy", "scalar", lambda d, _: d.entropy(), "entropy"),
]

_PROBABILITIES = (0.001, 0.05, 0.3, 0.5, 0.7, 0.95, 0.999)
"""The cdf levels the value grid sits at, so the points follow the mass at every `rate`."""


def _value_grid(rate: float) -> list[float]:
    """Below the support, a non-integral point, the support points at `_PROBABILITIES` and one where `sf ~ 1e-9`.

    `0` itself is `support_test.py`'s.
    """
    frozen = scipy_poisson(rate)
    points = {float(k) for k in frozen.ppf(_PROBABILITIES)} | {float(frozen.isf(1e-9))}
    return [-1.0, 2.5, *sorted(points)]


@pytest.mark.parametrize("rate", _RATES, ids=[f"rate={rate}" for rate in _RATES])
@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
def test_method_matches_scipy(case: Case[Poisson], rate: float) -> None:
    """Every method matches `scipy.stats.poisson(mu=rate)` across `_RATES`, the inverses as integers.

    Method-specific behaviour (null propagation, out-of-range quantiles, the support edge) is asserted
    in the shared contract files.
    """
    assert_case_matches_scipy(
        case,
        dist=Poisson(rate),
        scipy_frozen=scipy_poisson(rate),
        value_grid=_value_grid(rate),
        quantiles=_QUANTILES,
    )


# The tails past scipy's saturation. scipy's `logsf` and `logcdf` are the logs of its linear `sf` and `cdf`, so
# they are `-inf` where those underflow: `logsf(250, 5)` and `logcdf(10, 1000)` already are. Its `logpmf` stays
# accurate at these `rate`, so the oracle is the `logsumexp` of the masses the tail sums.


def _log_sf_oracle(k: int, rate: float) -> float:
    """`ln P(X > k)`; the masses above `k + 400` are below `e ** -2000` of the first one at these `rate`."""
    return float(logsumexp(scipy_poisson.logpmf(np.arange(k + 1, k + 401), rate)))


def _log_cdf_oracle(k: int, rate: float) -> float:
    """`ln P(X <= k)`."""
    return float(logsumexp(scipy_poisson.logpmf(np.arange(k + 1), rate)))


@pytest.mark.parametrize("k", [300, 1000])
def test_log_sf_past_the_linear_underflow(k: int) -> None:
    """`log_sf` stays finite and exact where `sf` is `0.0` and scipy's `logsf` is `-inf`."""
    got = pl.DataFrame({"k": [float(k)]}).select(r=Poisson(5.0).log_sf(pl.col("k")))["r"].item()
    assert got == pytest.approx(_log_sf_oracle(k, 5.0), rel=1e-12)


@pytest.mark.parametrize("k", [10, 100])
def test_log_cdf_deep_in_the_lower_tail(k: int) -> None:
    """`log_cdf` stays finite and exact far below the mean, where scipy's `logcdf` is `-inf` or near it."""
    got = pl.DataFrame({"k": [float(k)]}).select(r=Poisson(1000.0).log_cdf(pl.col("k")))["r"].item()
    assert got == pytest.approx(_log_cdf_oracle(k, 1000.0), rel=1e-12)


def test_isf_inverts_a_tail_mass_scipy_cannot_resolve() -> None:
    """`isf(1e-300)` is the smallest `k` with `sf(k) <= 1e-300`, solved against `q` rather than `ppf(1 - q)`."""
    rate, q = 5.0, 1e-300
    expected = next(k for k in range(1000) if _log_sf_oracle(k, rate) <= math.log(q))
    got = pl.DataFrame({"q": [q]}).select(r=Poisson(rate).isf(pl.col("q")))["r"].item()
    assert got == float(expected)


def test_ppf_resolves_a_quantile_deep_in_the_lower_tail() -> None:
    """`ppf(1e-300)` is the smallest `k` with `cdf(k) >= 1e-300`, nearly 600 below the mean at `rate = 1000`."""
    rate, q = 1000.0, 1e-300
    expected = next(k for k in range(1000) if _log_cdf_oracle(k, rate) >= math.log(q))
    got = pl.DataFrame({"q": [q]}).select(r=Poisson(rate).ppf(pl.col("q")))["r"].item()
    assert got == float(expected)


_ROUND_TRIP_RATES = [0.5, 4.5, 200.0]
_RESOLVED_COMPLEMENT = 1e-3
"""Below this tail mass the complement of a quantile near `1` no longer carries the step to the `1e-12` slack:
`1 - q` is exact, but `q` itself was rounded to `1.1e-16` absolute."""


@pytest.mark.parametrize("rate", _ROUND_TRIP_RATES, ids=lambda rate: f"rate={rate}")
def test_inverses_round_trip_every_support_point(rate: float) -> None:
    """`ppf(cdf(k)) == k` and `isf(sf(k)) == k`: a quantile computed *on* a step inverts to that step.

    The inverses' `1e-12` relative slack makes this hold: without it the last ulp of `cdf(k)`, read in log space,
    would decide between `k` and `k + 1`. Each direction is asserted where its input quantile still resolves the tail
    it is compared against.
    """
    ks = [float(k) for k in range(poisson_support_limit(rate))]
    dist = Poisson(rate)
    frame = (
        pl.DataFrame({"k": ks})
        .with_columns(cdf=dist.cdf(pl.col("k")), sf=dist.sf(pl.col("k")))
        .with_columns(ppf=dist.ppf(pl.col("cdf")), isf=dist.isf(pl.col("sf")))
    )
    ppf_rows = frame.filter(pl.col("sf") > _RESOLVED_COMPLEMENT)
    isf_rows = frame.filter(pl.col("cdf") > _RESOLVED_COMPLEMENT, pl.col("sf") > 0.0)
    assert ppf_rows.height > 0
    assert isf_rows.height > 0
    assert_series_equal(ppf_rows["ppf"], ppf_rows["k"], check_exact=True, check_names=False)
    assert_series_equal(isf_rows["isf"], isf_rows["k"], check_exact=True, check_names=False)


# NOTE: Deliberate divergences from scipy, recorded here rather than rediscovered as surprises.
#
# * Endpoints of the inverse: scipy's discrete `ppf(0)` is the point below the support, `-1`; this library
#   answers the smallest support point, `0`, the value every `q <= exp(-rate)` already inverts to. `ppf(1)` and
#   `isf(0)` are `inf` on both sides.
# * `rate = 0`: scipy accepts it as a point mass at `0`; this library raises, as statrs and `rand_distr` do.
# * The deep tails: scipy's `logsf` / `logcdf` are `-inf` once `sf` / `cdf` underflow, and its `isf` goes through
#   `ppf(1 - q)`; this library stays finite and inverts against `q` itself (asserted above against `logsumexp`
#   of scipy's own masses, since scipy cannot referee there directly).
# * The mass at a large `rate`: scipy's `logpmf` cancels to `1.1e-12` absolute at `rate = 1e3` and `2.5e-7` at
#   `1e8`. This library reads it off Stirling's form instead, so parity stops at `rate = 200` and
#   `tools/accuracy/audit.py` referees beyond it.
