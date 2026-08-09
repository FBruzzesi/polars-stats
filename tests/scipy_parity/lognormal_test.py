from __future__ import annotations

import math
import sys
from math import exp

import numpy as np
import polars as pl
import pytest
from scipy import special as scipy_special
from scipy.stats import lognorm as scipy_lognorm

from polars_stats import LogNormal
from tests.scipy_parity._harness import Case, assert_case_matches_scipy

# Parameter and evaluation grids for the parity sweep.
# Owned by this test category and independent of the per-method functional tests under `tests/distributions/lognormal`.
# `sigma` is kept moderate: the mean / variance grow exponentially in `sigma` and
# lose absolute precision against scipy past `sigma > 5`.
_PARAMS = [(0.0, 1.0), (0.5, 0.5), (-1.0, 0.25), (1.0, 0.75), (0.0, 0.1)]
_QUANTILES = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]

# Tolerance split, by upstream implementation rather than by aspiration:
#   * `statrs` evaluates pdf / ln_pdf to ~1e-15 against scipy, and the moments (exp-based) to
#     machine precision on this grid, so they hold the default 1e-12 target;
#   * cdf / sf (and their logs) and the closed-form ppf go through `erfc` / `erfc_inv`, while scipy
#     uses Cephes; the two agree only to ~1e-10. 1e-9 is the honest, comfortably-padded bound.
_TOL_ERF = 1e-9


def _scipy_frozen(mu: float, sigma: float) -> object:
    """`scipy.stats.lognorm` frozen at the polars-stats `(mu, sigma)`: `s=sigma`, `scale=exp(mu)`."""
    return scipy_lognorm(s=sigma, scale=exp(mu))


_CASES: list[Case[LogNormal]] = [
    Case("pdf", "value", lambda d, c: d.pdf(c), "pdf"),
    Case("log_pdf", "value", lambda d, c: d.log_pdf(c), "logpdf"),
    Case("cdf", "value", lambda d, c: d.cdf(c), "cdf", _TOL_ERF),
    Case("log_cdf", "value", lambda d, c: d.log_cdf(c), "logcdf", _TOL_ERF),
    Case("sf", "value", lambda d, c: d.sf(c), "sf", _TOL_ERF),
    Case("log_sf", "value", lambda d, c: d.log_sf(c), "logsf", _TOL_ERF),
    Case("ppf", "quantile", lambda d, c: d.ppf(c), "ppf", _TOL_ERF),
    Case("isf", "quantile", lambda d, c: d.isf(c), "isf", _TOL_ERF),
    Case("mean", "scalar", lambda d, _: d.mean(), "mean"),
    Case("variance", "scalar", lambda d, _: d.variance(), "var"),
    Case("std", "scalar", lambda d, _: d.std(), "std"),
    Case("median", "scalar", lambda d, _: d.median(), "median"),
    Case("entropy", "scalar", lambda d, _: d.entropy(), "entropy"),
]


def _value_grid(mu: float, sigma: float) -> list[float]:
    """Positive evaluation points, geometric around the median `exp(mu)`."""
    ks = pl.Series([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    return (mu + ks * sigma).exp().to_list()


@pytest.mark.parametrize(("mu", "sigma"), _PARAMS, ids=[f"mu={m},sigma={s}" for m, s in _PARAMS])
@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
def test_method_matches_scipy(case: Case[LogNormal], mu: float, sigma: float) -> None:
    """Every closed-form method matches `scipy.stats.lognorm(s=sigma, scale=exp(mu))` across the grid.

    Table-driven: adding a method is one row in `_CASES`. Method-specific behaviour (null
    propagation, out-of-range quantiles, the `x <= 0` support edge) is asserted in the per-method files.
    """
    assert_case_matches_scipy(
        case,
        dist=LogNormal(mu=mu, sigma=sigma),
        scipy_frozen=_scipy_frozen(mu, sigma),
        value_grid=_value_grid(mu, sigma),
        quantiles=_QUANTILES,
    )


# Standardised deviations of ln(x): `x = exp(mu + sigma * z)` probes both the far-right (large z)
# and far-left (x -> 0+, large negative z) tails where the naive `sf().log()` / `cdf().log()` are
# `-inf`, plus the direct-erfc band and the `ln_erfc` branch crossover (`z ~ 35.36`, bracketed by
# 35 and 36), mirroring the Normal grid.
_TAIL_HALF = (5.0, 10.0, 20.0, 30.0, 35.0, 36.0, 37.0, 38.0, 40.0, 50.0, 100.0)
_TAIL_Z = (*(-z for z in _TAIL_HALF), *_TAIL_HALF)


@pytest.mark.parametrize(("mu", "sigma"), [(0.0, 1.0), (1.0, 2.0)], ids=["mu=0,sigma=1", "mu=1,sigma=2"])
def test_log_tail_matches_scipy(mu: float, sigma: float) -> None:
    """`log_cdf` / `log_sf` stay finite and match scipy far into both tails, where `sf().log()` is `-inf`.

    Mirrors the Normal tail exemplar (the log-normal log-cdf / log-sf are the underlying normal's,
    composed with `ln`): probe *beyond* the underflow threshold, the regime the geometric
    `test_method_matches_scipy` grid never reaches.
    """
    xs = [exp(mu + sigma * z) for z in _TAIL_Z]
    dist = LogNormal(mu=mu, sigma=sigma)
    frozen = scipy_lognorm(s=sigma, scale=exp(mu))

    got = pl.DataFrame({"x": xs}).select(log_cdf=dist.log_cdf("x"), log_sf=dist.log_sf("x"))
    assert got["log_cdf"].is_finite().all(), "log_cdf underflowed to -inf in the tail"
    assert got["log_sf"].is_finite().all(), "log_sf underflowed to -inf in the tail"
    np.testing.assert_allclose(got["log_cdf"].to_numpy(), frozen.logcdf(xs), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(got["log_sf"].to_numpy(), frozen.logsf(xs), rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize(("mu", "sigma"), [(0.0, 1.0), (1.0, 2.0)], ids=["mu=0,sigma=1", "mu=1,sigma=2"])
def test_log_near_one_side_keeps_relative_precision(mu: float, sigma: float) -> None:
    """The near-certain side matches scipy in *relative* terms (``atol=0``), not just within slack.

    Mirrors the Normal pin of the ``log1p`` branch: ``log_cdf`` far right of the median (and
    ``log_sf`` far left) is a tiny negative number the erfc reflection would round to ``0.0``.
    """
    zs = [1.0, 5.0, 10.0, 20.0, 30.0]
    dist = LogNormal(mu=mu, sigma=sigma)
    frozen = scipy_lognorm(s=sigma, scale=exp(mu))

    upper = [exp(mu + sigma * z) for z in zs]
    lower = [exp(mu - sigma * z) for z in zs]
    got = pl.DataFrame({"hi": upper, "lo": lower}).select(
        log_cdf=dist.log_cdf(pl.col("hi")),
        log_sf=dist.log_sf(pl.col("lo")),
    )
    np.testing.assert_allclose(got["log_cdf"].to_numpy(), frozen.logcdf(upper), rtol=1e-9, atol=0.0)
    np.testing.assert_allclose(got["log_sf"].to_numpy(), frozen.logsf(lower), rtol=1e-9, atol=0.0)


# The `isf` regression. `isf` was the base-class `ppf(1 - quantile)` until X5; it is now the
# underlying normal's symmetry form exponentiated, which forms no complement.
#
# Oracled by `scipy.special.ndtri` under the symmetry `z_(1-q) = -z_q`, **not** by
# `scipy.stats.lognorm.isf`: scipy composes the same way, so it carries the same defect and cannot
# referee it. Composing through `exp` turns the normal's absolute quantile error into a relative one
# here, which is why a large `sigma` made this the worst `isf` in the library (`1.1e-07` at
# `LogNormal(0, 20).isf(1e-9)`).
_ISF_DEEP_QUANTILES = [1e-300, 1e-100, 1e-40, 1e-16, 1e-9, 1e-8, 1e-4, 0.3, 0.5, 0.9]


@pytest.mark.parametrize(("mu", "sigma"), [(0.0, 1.0), (0.0, 20.0), (3.0, 2.0), (-5.0, 0.25)], ids=str)
def test_isf_keeps_relative_precision_across_300_decades(mu: float, sigma: float) -> None:
    """`isf` holds full relative precision arbitrarily deep, where it used to degrade as `1.1e-16 / q`.

    Asserted in log space, with an *absolute* tolerance: the values here span 600 decades, and an
    absolute error in the log is exactly a relative error in the value, so `atol=1e-12` on the log
    states the `1e-12` relative claim uniformly. Comparing the values directly would also overflow
    the oracle at `sigma = 20`, where the true `isf(1e-300)` exceeds float64 and `inf` is correct.
    """
    got = pl.DataFrame({"q": _ISF_DEEP_QUANTILES}).select(r=LogNormal(mu=mu, sigma=sigma).isf(pl.col("q")))["r"]
    expected_log = mu - sigma * scipy_special.ndtri(np.array(_ISF_DEEP_QUANTILES))
    representable = expected_log < math.log(sys.float_info.max)
    assert np.isfinite(got.to_numpy()[representable]).all(), "isf saturated where the answer is representable"
    np.testing.assert_allclose(np.log(got.to_numpy()[representable]), expected_log[representable], rtol=0.0, atol=1e-12)


# The `std` regression: `std` inherited `variance().sqrt()`, and so inherited an overflow the square
# root would have undone. The variance genuinely exceeds float64 above `sigma ~ 18.8`, but the
# standard deviation only does above `sigma ~ 26.6`.
#
# Oracled by hard-coded 50-digit `mpmath` values, not by scipy: `scipy.stats.lognorm.std` composes
# through the variance too and returns `inf` at every one of these points.
_LOGSPACE_STD_CASES = [
    (0.0, 19.0, 6.0298702490003524e156),
    (0.0, 20.0, 5.221469689764144e173),
    (2.0, 25.0, 2.0074288128646431e272),
]


@pytest.mark.parametrize(("mu", "sigma", "expected"), _LOGSPACE_STD_CASES, ids=lambda v: str(v)[:10])
def test_std_survives_where_the_variance_overflows(mu: float, sigma: float, expected: float) -> None:
    """`std` is finite and correct where `variance()` is legitimately `inf`."""
    frame = pl.DataFrame({"_": [0]})
    assert math.isinf(frame.select(r=LogNormal(mu=mu, sigma=sigma).variance())["r"].item()), (
        "variance no longer overflows here, so this case no longer tests anything"
    )
    got = frame.select(r=LogNormal(mu=mu, sigma=sigma).std())["r"].item()
    assert math.isfinite(got), "std inherited the variance overflow"
    assert got == pytest.approx(expected, rel=1e-14)


# The `variance` regression, same identity from the other end: `exp(sigma**2) - 1` cancels for a
# small `sigma`, losing 8 of 16 digits at `sigma = 1e-4`. Spelled `2 * exp(t/2) * sinh(t/2)` instead,
# which is `expm1(t)` exactly (polars has no `expm1`). Oracled by `math.expm1`, which does have one.
@pytest.mark.parametrize("sigma", [1e-6, 1e-4, 1e-3, 0.01, 0.5])
def test_variance_and_std_keep_precision_for_a_tiny_sigma(sigma: float) -> None:
    """The moments hold their `1e-12` claim where the naive difference of exponentials cancels."""
    got = pl.DataFrame({"_": [0]}).select(
        variance=LogNormal(mu=0.0, sigma=sigma).variance(),
        std=LogNormal(mu=0.0, sigma=sigma).std(),
    )
    expected_variance = math.expm1(sigma**2) * math.exp(sigma**2)
    assert got["variance"].item() == pytest.approx(expected_variance, rel=1e-14)
    assert got["std"].item() == pytest.approx(math.sqrt(expected_variance), rel=1e-14)
