from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy import special as scipy_special
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
        dist=Normal(mu=mean, sigma=std),
        scipy_frozen=scipy_norm(loc=mean, scale=std),
        value_grid=_value_grid(mean, std),
        quantiles=_QUANTILES,
    )


# Deviations (in sigma units) spanning the direct-erfc band, the `ln_erfc` branch crossover
# (`t = 25`, i.e. `z = 25 * sqrt(2) ~ 35.36`, bracketed by 35 and 36), and the far tail past the
# ~38 sigma point where `sf` / `cdf` underflow to `0` and the naive `sf().log()` / `cdf().log()`
# return `-inf`. Both signs, so each method covers its active tail and the near-one opposite side.
_TAIL_HALF = (5.0, 10.0, 20.0, 30.0, 35.0, 36.0, 37.0, 38.0, 40.0, 50.0, 100.0)
_TAIL_Z = (*(-z for z in _TAIL_HALF), *_TAIL_HALF)


@pytest.mark.parametrize(("mean", "std"), [(0.0, 1.0), (2.0, 3.0)], ids=["std-normal", "mean=2,std=3"])
def test_log_tail_matches_scipy(mean: float, std: float) -> None:
    """`log_cdf` / `log_sf` stay finite and match scipy far into the tails, where `sf().log()` is `-inf`.

    The regime C1 targets (anomaly scoring on many-sigma events) and the exemplar every future
    distribution with a genuine tail must reproduce: probe *beyond* the underflow threshold, not
    merely `sf ~ 1e-300` where the naive path is still finite. `test_method_matches_scipy` stops at
    3 sigma and never reaches here.
    """
    xs = [mean + std * z for z in _TAIL_Z]
    dist = Normal(mu=mean, sigma=std)
    frozen = scipy_norm(loc=mean, scale=std)

    got = pl.DataFrame({"x": xs}).select(
        log_cdf=dist.log_cdf(pl.col("x")),
        log_sf=dist.log_sf(pl.col("x")),
    )
    assert got["log_cdf"].is_finite().all(), "log_cdf underflowed to -inf in the tail"
    assert got["log_sf"].is_finite().all(), "log_sf underflowed to -inf in the tail"
    np.testing.assert_allclose(got["log_cdf"].to_numpy(), frozen.logcdf(xs), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(got["log_sf"].to_numpy(), frozen.logsf(xs), rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize(("mean", "std"), [(0.0, 1.0), (2.0, 3.0)], ids=["std-normal", "mean=2,std=3"])
def test_log_near_one_side_keeps_relative_precision(mean: float, std: float) -> None:
    """The near-certain side matches scipy in *relative* terms (``atol=0``), not just within slack.

    ``log_cdf`` far above the mean (and ``log_sf`` far below) is a tiny negative number, e.g.
    ``-7.6e-24`` at 10 sigma; the erfc reflection ``2 - erfc(t)`` would round it to exactly
    ``0.0``, an error invisible to the ``atol``-padded tail test above. Pins the ``log1p`` branch.
    """
    zs = [1.0, 5.0, 10.0, 20.0, 30.0]
    dist = Normal(mu=mean, sigma=std)
    frozen = scipy_norm(loc=mean, scale=std)

    upper = [mean + std * z for z in zs]
    lower = [mean - std * z for z in zs]
    got = pl.DataFrame({"hi": upper, "lo": lower}).select(
        log_cdf=dist.log_cdf(pl.col("hi")),
        log_sf=dist.log_sf(pl.col("lo")),
    )
    np.testing.assert_allclose(got["log_cdf"].to_numpy(), frozen.logcdf(upper), rtol=1e-9, atol=0.0)
    np.testing.assert_allclose(got["log_sf"].to_numpy(), frozen.logsf(lower), rtol=1e-9, atol=0.0)


# The `isf` regression. `isf` was the base-class `ppf(1 - quantile)` until X5; it is now
# `mu + sigma * sqrt(2) * erfc_inv(2q)`, which forms no complement.
#
# Oracled by `scipy.special.ndtri` (the inverse standard-normal cdf) under the normal's symmetry
# `z_(1-q) = -z_q`, **not** by `scipy.stats.norm.isf`: scipy's `isf` is `ppf(1 - q)` too, so it
# carries the same defect and cannot referee it. `ndtri` at a small `q` is the accurate branch of an
# independent Cephes implementation.
_ISF_DEEP_QUANTILES = [1e-300, 1e-100, 1e-40, 1e-16, 1e-9, 1e-8, 1e-4, 0.3, 0.5, 0.9]


@pytest.mark.parametrize(("mean", "std"), _PARAMS, ids=str)
def test_isf_keeps_relative_precision_across_300_decades(mean: float, std: float) -> None:
    """`isf` holds full relative precision arbitrarily deep, where it used to degrade as `1.1e-16 / q`.

    Worst measured case was `Normal(-3, 0.5).isf(1e-9)`: relatively wrong by `2.09e-06` against a
    50-digit oracle, because the answer *also* nearly cancels there (`mu` and `sigma * z` agree to
    three digits), so the quantile's absolute error is amplified on the way out. Now `2.6e-13`.
    """
    got = pl.DataFrame({"q": _ISF_DEEP_QUANTILES}).select(r=Normal(mu=mean, sigma=std).isf(pl.col("q")))["r"]
    expected = mean - std * scipy_special.ndtri(np.array(_ISF_DEEP_QUANTILES))
    assert got.is_finite().all()
    np.testing.assert_allclose(got.to_numpy(), expected, rtol=1e-11, atol=0.0)


def test_isf_beats_scipy_below_the_complement_resolution() -> None:
    """Pins that we are *not* reproducing `scipy.stats.norm.isf`, which saturates below `q ~ 1e-17`.

    `1 - q` rounds to exactly `1.0` there, so scipy's composition returns `inf` for every quantile
    that deep while the true answer is an ordinary finite number. A parity test written against
    scipy would have locked in the defect; this asserts the disagreement on purpose.
    """
    deep = [1e-20, 1e-100, 1e-300]
    got = pl.DataFrame({"q": deep}).select(r=Normal().isf(pl.col("q")))["r"]
    assert got.is_finite().all(), "isf saturated the way scipy does"
    np.testing.assert_allclose(got.to_numpy(), -scipy_special.ndtri(np.array(deep)), rtol=1e-12, atol=0.0)
