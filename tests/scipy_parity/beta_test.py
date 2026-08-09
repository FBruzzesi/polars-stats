from __future__ import annotations

import math
import sys
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from scipy import special as scipy_special
from scipy.stats import beta as scipy_beta

from polars_stats import Beta
from tests.scipy_parity._harness import Case, assert_case_matches_scipy

if TYPE_CHECKING:
    from collections.abc import Callable

# Parameter and evaluation grids for the parity sweep. Owned by this test category and independent
# of the per-method functional tests under `tests/distributions/beta`. The grid covers the distinct
# shape regimes: unimodal, U-shaped (both shapes < 1), uniform (1, 1), J-shaped, skewed, and the
# large-shape regime where statrs switches its pdf to `ln_pdf().exp()`.
_PARAMS = [(2.0, 3.0), (0.5, 0.5), (1.0, 1.0), (5.0, 1.0), (2.0, 8.0), (90.0, 100.0)]
_QUANTILES = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
# The support is fixed to [0, 1], so one interior grid spanning both tails serves every shape pair.
_VALUE_GRID = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]

# Tolerance split, by upstream implementation rather than by aspiration:
#   * cdf / sf (and their logs), ppf / isf (AS 64 + Newton inverse), the moments and the entropy
#     agree with scipy's Cephes incomplete-beta family to ~1e-13 across the grid, holding the
#     default 1e-12 target;
#   * pdf goes through `ln_pdf().exp()` in statrs for shapes > 80, which costs a few digits of
#     absolute accuracy at the large-shape pair (~1e-12 observed). 1e-11 is the honest, padded bound.
_TOL_LARGE_SHAPE_PDF = 1e-11


_CASES: list[Case[Beta]] = [
    Case("pdf", "value", lambda d, c: d.pdf(c), "pdf", _TOL_LARGE_SHAPE_PDF),
    Case("log_pdf", "value", lambda d, c: d.log_pdf(c), "logpdf"),
    Case("cdf", "value", lambda d, c: d.cdf(c), "cdf"),
    Case("log_cdf", "value", lambda d, c: d.log_cdf(c), "logcdf"),
    Case("sf", "value", lambda d, c: d.sf(c), "sf"),
    Case("log_sf", "value", lambda d, c: d.log_sf(c), "logsf"),
    Case("ppf", "quantile", lambda d, c: d.ppf(c), "ppf"),
    Case("isf", "quantile", lambda d, c: d.isf(c), "isf"),
    Case("mean", "scalar", lambda d, _: d.mean(), "mean"),
    Case("variance", "scalar", lambda d, _: d.variance(), "var"),
    Case("std", "scalar", lambda d, _: d.std(), "std"),
    Case("median", "scalar", lambda d, _: d.median(), "median"),
    Case("entropy", "scalar", lambda d, _: d.entropy(), "entropy"),
]


@pytest.mark.parametrize(("a", "b"), _PARAMS, ids=[f"a={a},b={b}" for a, b in _PARAMS])
@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
def test_method_matches_scipy(case: Case[Beta], a: float, b: float) -> None:
    """Every closed-form method matches `scipy.stats.beta` across the `(a, b)` grid.

    Table-driven: adding a method is one row in `_CASES`. Method-specific behaviour (null
    propagation, out-of-range quantiles, the support boundaries) is asserted in the per-method files.
    """
    assert_case_matches_scipy(
        case,
        dist=Beta(a=a, b=b),
        scipy_frozen=scipy_beta(a=a, b=b),
        value_grid=_VALUE_GRID,
        quantiles=_QUANTILES,
    )


def _series_ln_beta_cdf(a: float, b: float, xs: list[float]) -> np.ndarray:
    """Reference `ln I_x(a, b)` in the deep left corner, where scipy's `logcdf` is already `-inf`.

    The hypergeometric representation `I_x(a, b) = x^a / (a B(a, b)) * 2F1(a, 1 - b; a + 1; x)`
    (DLMF 8.17.7), taken in logs: the `2F1` factor is `O(1)` and cancellation-free for `b x <= 1`
    (the callers restrict themselves to that domain; beyond it the alternating polynomial loses
    digits), so only the prefactor needs log space. Independent of the implementation under test,
    which evaluates the Lentz continued fraction, not this series. `x = 0` maps to `-inf` exactly.
    """
    xs_arr = np.asarray(xs)
    with np.errstate(divide="ignore"):
        return (
            a * np.log(xs_arr)
            - math.log(a)
            - scipy_special.betaln(a, b)
            + np.log(scipy_special.hyp2f1(a, 1.0 - b, a + 1.0, xs_arr))
        )


# Per-pair grids: each spans the scipy-finite band and crosses scipy's underflow point (the log
# prefactor below ~-745), so both the parity band and the beyond-scipy band are exercised for
# every pair.
_LEFT_CORNER_CASES = [
    (200.0, 2.0, [1e-200, 1e-4, 0.01, 0.05, 0.3, 0.5]),
    (80.0, 90.0, [1e-200, 1e-4, 0.01, 0.05, 0.2, 0.4]),
    (2.0, 8.0, [1e-200, 1e-8, 1e-4, 0.01, 0.1]),
]


@pytest.mark.parametrize(("a", "b", "xs"), _LEFT_CORNER_CASES, ids=lambda v: str(v)[:16])
def test_log_cdf_left_corner_matches_scipy_or_reference(a: float, b: float, xs: list[float]) -> None:
    """`log_cdf` stays finite and accurate deep in the left corner, where `cdf().log()` is `-inf`.

    scipy's `beta.logcdf` is itself naive (`-inf` in the same regime, no stable log-betainc in
    scipy), so scipy is the oracle only where it is finite; past its underflow the oracle is the
    independent log-space hypergeometric form, applied on its cancellation-free domain `b x <= 1`
    (which provably contains every beyond-scipy point: underflow needs `a ln x < ~-700`, hence a
    tiny `x`).
    """
    dist = Beta(a=a, b=b)
    frozen = scipy_beta(a=a, b=b)

    got = pl.DataFrame({"x": xs}).select(log_cdf=dist.log_cdf(pl.col("x")))["log_cdf"].to_numpy()
    assert np.isfinite(got).all(), "log_cdf underflowed to -inf in the left corner"

    ref = frozen.logcdf(xs)
    finite = np.isfinite(ref)
    series_safe = np.asarray([b * x <= 1.0 for x in xs], dtype=bool)
    assert finite.any(), "corner grid never reaches the scipy-finite band"
    assert (~finite).any(), "corner grid never crosses scipy's underflow point"
    assert (finite | series_safe).all(), "a beyond-scipy point fell outside the series oracle's domain"

    np.testing.assert_allclose(got[finite], ref[finite], rtol=1e-10, atol=1e-12)
    series_xs = [x for x in xs if b * x <= 1.0]
    np.testing.assert_allclose(got[series_safe], _series_ln_beta_cdf(a, b, series_xs), rtol=1e-11, atol=1e-12)


# The right-corner counterpart, evaluated at `1 - y`: `sf(1 - y) = I_y(b, a)`, so the underflow
# driver is `b ln y` and the pairs need a large `b` to cross it at all (the complement `1 - y`
# rounds to `1.0` below `y ~ 1e-16`, so `y` cannot go arbitrarily small the way the left grids'
# `1e-200` does; `b = 2`, say, can never underflow on this side).
_RIGHT_CORNER_CASES = [
    (2.0, 200.0, [1e-4, 0.01, 0.05, 0.3, 0.5]),
    (90.0, 80.0, [1e-16, 1e-4, 0.01, 0.05, 0.2, 0.4]),
    (8.0, 200.0, [1e-4, 0.01, 0.05, 0.1, 0.3]),
]


@pytest.mark.parametrize(("a", "b", "ys"), _RIGHT_CORNER_CASES, ids=lambda v: str(v)[:16])
def test_log_sf_right_corner_matches_scipy_or_reference(a: float, b: float, ys: list[float]) -> None:
    """`log_sf` stays finite and accurate deep in the right corner, where `sf().log()` is `-inf`.

    Structure and oracles of the left-corner test with the `I_x` symmetry applied
    (`sf(1 - y) = I_y(b, a)`). The series oracle is evaluated at the complement the implementation
    itself recomputes from the rounded input (`1 - (1 - y)`), so the comparison tests the algorithm
    rather than the half-ulp the input rounding costs below `y = 0.5`.
    """
    vs = [1.0 - y for y in ys]
    y_eff = [1.0 - v for v in vs]
    dist = Beta(a=a, b=b)
    frozen = scipy_beta(a=a, b=b)

    got = pl.DataFrame({"v": vs}).select(log_sf=dist.log_sf(pl.col("v")))["log_sf"].to_numpy()
    assert np.isfinite(got).all(), "log_sf underflowed to -inf in the right corner"

    ref = frozen.logsf(vs)
    finite = np.isfinite(ref)
    series_safe = np.asarray([a * y <= 1.0 for y in y_eff], dtype=bool)
    assert finite.any(), "corner grid never reaches the scipy-finite band"
    assert (~finite).any(), "corner grid never crosses scipy's underflow point"
    assert (finite | series_safe).all(), "a beyond-scipy point fell outside the series oracle's domain"

    np.testing.assert_allclose(got[finite], ref[finite], rtol=1e-10, atol=1e-12)
    series_ys = [y for y in y_eff if a * y <= 1.0]
    np.testing.assert_allclose(got[series_safe], _series_ln_beta_cdf(b, a, series_ys), rtol=1e-11, atol=1e-12)


# The near-certain side is a tiny negative number whose *relative* precision the `ln_1p` reflection
# branches must keep; points sit where the opposite tail is small but far above underflow, so
# scipy's linear `sf` / `cdf` carry full relative precision as the oracle.
_NEAR_CERTAIN_CASES = [
    (2.0, 3.0, [0.9, 0.99, 1.0 - 1e-9], [1e-9, 1e-3, 0.05]),
    (90.0, 100.0, [0.6, 0.65, 0.7], [0.3, 0.35, 0.4]),
]


@pytest.mark.parametrize(("a", "b", "upper", "lower"), _NEAR_CERTAIN_CASES, ids=lambda v: str(v)[:16])
def test_log_near_certain_side_keeps_relative_precision(
    a: float, b: float, upper: list[float], lower: list[float]
) -> None:
    """`log_cdf` near `1` and `log_sf` near `0` keep relative precision (``atol=0``).

    scipy's naive `logcdf` / `logsf` cannot be the oracle here (`log(1 - s)` on the rounded linear
    value loses relative precision once `s` nears the double rounding), but its *linear*
    complements are exact where small, so `log1p` of them is.
    """
    dist = Beta(a=a, b=b)
    frozen = scipy_beta(a=a, b=b)

    got = pl.DataFrame({"hi": upper, "lo": lower}).select(
        log_cdf=dist.log_cdf(pl.col("hi")),
        log_sf=dist.log_sf(pl.col("lo")),
    )
    np.testing.assert_allclose(got["log_cdf"].to_numpy(), np.log1p(-frozen.sf(upper)), rtol=1e-11, atol=0.0)
    np.testing.assert_allclose(got["log_sf"].to_numpy(), np.log1p(-frozen.cdf(lower)), rtol=1e-11, atol=0.0)


# The `ppf` regressions found by `make audit`. Each was a defect of the `statrs` AS 64 inverse this
# crate no longer calls (see `beta.rs::inverse_cdf`), and each is oracled by a closed form rather
# than by scipy, which is itself unreliable that deep.
#
# `Beta(a, 1)` and `Beta(1, b)` have elementary cdfs (`x**a` and `1 - (1 - x)**b`), so their inverses
# are exact for *every* quantile, including ones no special-function reference resolves.
def _power_ppf(a: float) -> Callable[[float], float]:
    """Exact `ppf` for `Beta(a, 1)`, whose cdf is `x**a`."""
    return lambda q: q ** (1.0 / a)


def _complement_power_ppf(b: float) -> Callable[[float], float]:
    """Exact `ppf` for `Beta(1, b)`, whose cdf is `1 - (1 - x)**b`."""
    return lambda q: -math.expm1(math.log1p(-q) / b)


def _arcsine_ppf(q: float) -> float:
    """Exact `ppf` for `Beta(0.5, 0.5)`: the arcsine law's `cdf(x) = (2 / pi) asin(sqrt(x))`."""
    return math.sin(q * math.pi / 2) ** 2


_CLOSED_FORM_PPF_CASES: list[tuple[float, float, Callable[[float], float]]] = [
    (3.0, 1.0, _power_ppf(3.0)),
    (0.25, 1.0, _power_ppf(0.25)),
    (1.0, 4.0, _complement_power_ppf(4.0)),
    (0.5, 0.5, _arcsine_ppf),
]
_DEEP_QUANTILES = [1e-300, 1e-165, 1e-100, 1e-40, 1e-10, 1e-4, 0.3, 0.5, 0.9, 1.0 - 1e-9]


@pytest.mark.parametrize(("a", "b", "exact"), _CLOSED_FORM_PPF_CASES, ids=lambda v: str(v)[:12])
def test_ppf_matches_closed_form_into_the_deep_tail(a: float, b: float, exact: Callable[[float], float]) -> None:
    """`ppf` is accurate across 300 decades of `q`, on shape pairs whose inverse is elementary.

    Three separate audit findings live here, all in `q` regimes no curated grid had reached:
    `statrs`' inverse *panicked* below `q ~ 1e-165` (an unguarded Newton step left `[0, 1]` and
    `beta_reg` unwrapped an `XOutOfRange`, aborting the whole query), did not terminate for `q` in
    ~`[1e-150, 1e-60]` at some shapes, and returned one constant across 100 decades because its
    convergence floor was absolute rather than relative.
    """
    got = pl.DataFrame({"q": _DEEP_QUANTILES}).select(ppf=Beta(a=a, b=b).ppf(pl.col("q")))["ppf"].to_numpy()
    expected = np.array([exact(q) for q in _DEEP_QUANTILES])
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, expected, rtol=1e-11, atol=0.0)


# The inverse is routed by *where the root is* (the mass at `x = 0.5`), not by which tail mass is
# smaller. Routing on `q <= 0.5` parts company with the root's location as soon as the shapes are
# skewed: `Beta(0.001, 1)` puts 99.93% of its mass below `x = 0.5`, so every `q` in `(0.5, 0.9993]`
# went to the upper solve, whose variable `ln(1 - x)` has only absolute resolution near `0` and
# cannot represent a root below `~1e-16`. All of them answered one constant, `6.5e-16`, against a
# true `ppf(0.7) = 0.7 ** 1000 ~ 1.3e-155`.
#
# These shapes are the regime the closed-form cases above miss: `a = 0.25` is the smallest there, and
# its median is `0.0625`, ten decades above where the upper solve stops resolving.
_SKEWED_PPF_CASES = [(0.001, 1.0), (0.01, 1.0), (0.05, 1.0)]
_ABOVE_MEDIAN_QUANTILES = [0.5, 0.5 + 1e-16, 0.51, 0.6, 0.7, 0.9, 0.99, 1.0 - 1e-9]


@pytest.mark.parametrize(("a", "b"), _SKEWED_PPF_CASES, ids=str)
def test_ppf_resolves_a_root_far_below_the_median_mass(a: float, b: float) -> None:
    """`ppf` stays exact above `q = 0.5` when the root is still deep in the left corner.

    `Beta(a, 1)` has the exact inverse `q ** (1 / a)`, so this needs no oracle at all. It is also
    monotone by construction, which the assertion checks separately: the failure mode was a run of
    identical answers, which a per-point relative check catches but a shape check would not.
    """
    got = pl.DataFrame({"q": _ABOVE_MEDIAN_QUANTILES}).select(r=Beta(a=a, b=b).ppf(pl.col("q")))["r"].to_numpy()
    expected = np.array([q ** (1.0 / a) for q in _ABOVE_MEDIAN_QUANTILES])
    np.testing.assert_allclose(got, expected, rtol=1e-11, atol=0.0)
    assert np.all(np.diff(got) > 0.0), "ppf must be strictly increasing across the routing crossover"


@pytest.mark.parametrize(("a", "b"), _SKEWED_PPF_CASES, ids=str)
def test_isf_resolves_a_root_far_below_the_median_mass(a: float, b: float) -> None:
    """The `isf` mirror of the routing regression: `isf(q) = ppf(1 - q)` must pick the same branch."""
    quantiles = [1.0 - q for q in _ABOVE_MEDIAN_QUANTILES]
    got = pl.DataFrame({"q": quantiles}).select(r=Beta(a=a, b=b).isf(pl.col("q")))["r"].to_numpy()
    expected = np.array([q ** (1.0 / a) for q in _ABOVE_MEDIAN_QUANTILES])
    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=0.0)


# `I_0.5(s, s) = 1/2` exactly, at every shape `s`, by symmetry. A free exact oracle at parameters
# where scipy is the only other reference and mpmath is slow, and the one that exposes the continued
# fraction's iteration cap: statrs (and Numerical Recipes) stop at 141 terms, but the Lentz
# recurrence needs ~`sqrt(a + b)` of them, so above `s ~ 1e4` it truncates silently. That returned a
# *positive* `log_cdf` (`+0.76` at `s = 1e8`, a probability above 1) and a negative `cdf` (`-1.147`).
_SYMMETRIC_SHAPES = [1e2, 1e3, 1e4, 1e5, 1e6, 1e8]


@pytest.mark.parametrize("shape", _SYMMETRIC_SHAPES, ids=lambda s: f"s={s:.0e}")
def test_log_tails_hold_the_symmetry_identity_at_large_shapes(shape: float) -> None:
    """`log_cdf(0.5) == log_sf(0.5) == -log 2` for `Beta(s, s)`, to a shape-dependent bound.

    The tolerance is derived, not chosen. Once the continued fraction runs to convergence the
    residual is the cancellation in `ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b)`, which is
    `~eps * ln_gamma(a + b)` absolute in the log and therefore grows with the shape: `2.8e-09` at
    `s = 1e6`, `7.1e-07` at `1e8`. `8 * eps` carries the observed constant with margin. That is the
    bound the README accuracy notes quote, and no iteration count removes it.

    The sign assertion is the part that must never relax: a positive `log_cdf` is a probability above
    `1`, which is what the truncated 141-term fraction returned (`+0.76` at `s = 1e8`).
    """
    dist = Beta(a=shape, b=shape)
    tolerance = 8 * sys.float_info.epsilon * math.lgamma(2 * shape)
    got = pl.DataFrame({"x": [0.5]}).select(lc=dist.log_cdf(pl.col("x")), ls=dist.log_sf(pl.col("x")))
    for name in ("lc", "ls"):
        value = got[name].item()
        assert value < 0.0, f"{name} is a log-probability and cannot be positive"
        assert value == pytest.approx(-math.log(2.0), rel=0.0, abs=tolerance), name


@pytest.mark.parametrize("q", [1e-300, 1e-200, 1e-165, 1e-120])
def test_ppf_does_not_abort_the_query_in_the_deep_tail(q: float) -> None:
    """A deep quantile returns a value rather than aborting the query.

    Kept separate from the accuracy test because the failure mode was categorically different: a
    `panic` inside `statrs` surfaces as an opaque `ComputeError` that takes down every other row in
    the frame, so no amount of tolerance would have caught it.
    """
    frame = pl.DataFrame({"q": [q]})
    for a, b in [(2.0, 3.0), (200.0, 2.0), (0.05, 0.05), (80.0, 90.0), (1000.0, 1000.0)]:
        value = frame.select(r=Beta(a=a, b=b).ppf(pl.col("q")))["r"].item()
        assert math.isfinite(value), f"Beta({a}, {b}).ppf({q}) is not finite"
        assert 0.0 <= value <= 1.0


@pytest.mark.parametrize(("a", "b"), [(0.05, 0.05), (0.1, 500.0)])
@pytest.mark.parametrize("x", [1e-16, 1e-22, 1e-57])
def test_sf_keeps_the_lower_corner_for_small_shapes(a: float, b: float, x: float) -> None:
    """`sf` at a tiny `x` with `a < 1`, where `statrs`' own `sf` returns a flat `1.0`.

    `statrs` evaluates `sf` as `beta_reg(b, a, 1 - x)`, passing the complement as the *argument*: it
    rounds to exactly `1` below `x ~ 1e-16` and the whole answer is lost (`Beta(0.05, 0.05).sf(1e-16)`
    came back `1.0` against a true `0.9205`). scipy is a valid oracle on these pairs, the value
    being `O(1)` and far from `1`.
    """
    got = pl.DataFrame({"x": [x]}).select(r=Beta(a=a, b=b).sf(pl.col("x")))["r"].item()
    assert got == pytest.approx(float(scipy_beta.sf(x, a, b)), rel=1e-12)


@pytest.mark.parametrize("x", [1e-16, 1e-22, 1e-57])
def test_sf_beats_scipy_in_the_arcsine_lower_corner(x: float) -> None:
    """The same corner on `Beta(0.5, 0.5)`, where scipy makes the mistake statrs made.

    `scipy.stats.beta(0.5, 0.5).sf(1e-16)` is relatively wrong by ~`6e-10` (and saturates entirely
    further down), so it cannot be the oracle. The arcsine law gives the exact value in closed form,
    `1 - (2 / pi) asin(sqrt(x))`, which is why this case is split out of the scipy-parity one above
    rather than folded into it behind a loose tolerance.
    """
    got = pl.DataFrame({"x": [x]}).select(r=Beta(a=0.5, b=0.5).sf(pl.col("x")))["r"].item()
    expected = 1.0 - (2.0 / math.pi) * math.asin(math.sqrt(x))
    assert got == pytest.approx(expected, rel=1e-15)


# The `isf` regressions. `isf` was the base-class `ppf(1 - quantile)` until X5; it now enters the
# same bounded solve from the upper tail (`beta.rs::inverse_sf`), so it never forms a complement.
#
# Oracled by the round-trip against `log_sf` rather than by a closed form: the deep-`isf` answer is
# always *near 1*, where float64 has no resolution left and every closed form cancels as badly as the
# implementation being tested. `log_sf` has no such problem (it is finite and accurate into both
# corners via Port B, and is itself audited), and inverting the audited forward function is the
# property `ppf(1 - q)` actually violated.
# Split by whether the *answer* is resolvable, not by shape aesthetics. A deep `isf` sits near the
# upper support edge, and how near depends on `b`: the answer approaches `1` like `q ** (1 / b)`, so
# `Beta(0.5, 0.5).isf(1e-16)` is `1.0` exactly and correctly, no float64 lying between. Round-tripping
# a saturated answer measures nothing (`log_sf(1.0)` is `-inf` by definition), so those pairs get the
# saturation test instead. A large `b` keeps the answer well inside the support for 300 decades.
_ISF_RESOLVING_PARAMS = [(0.1, 500.0), (2.0, 200.0), (90.0, 100.0)]
_ISF_SATURATING_PARAMS = [(2.0, 3.0), (0.5, 0.5), (5.0, 1.0)]
_ISF_DEEP_QUANTILES = [1e-300, 1e-100, 1e-40, 1e-16, 1e-9, 1e-4, 0.3, 0.5, 0.9]


@pytest.mark.parametrize(("a", "b"), _ISF_RESOLVING_PARAMS, ids=str)
def test_isf_inverts_log_sf_across_300_decades(a: float, b: float) -> None:
    """`log_sf(isf(q)) == log(q)` across the full quantile range.

    Composing `ppf(1 - quantile)` quantised the tail mass to the `1.1e-16` resolution of the
    complement before the solve ever ran, so this round-trip degraded as `1.1e-16 / q` and was
    meaningless below `q ~ 1e-16`. Measured at `Beta(0.1, 500).isf(1e-9)`, the value itself was
    relatively wrong by `1.66e-09` against a 50-digit oracle; it is now `~1e-14`.
    """
    frame = pl.DataFrame({"q": _ISF_DEEP_QUANTILES})
    dist = Beta(a=a, b=b)
    got = frame.select(r=dist.log_sf(dist.isf(pl.col("q"))))["r"]
    assert got.is_finite().all(), "isf saturated where the answer is representable"
    np.testing.assert_allclose(got.to_numpy(), np.log(np.array(_ISF_DEEP_QUANTILES)), rtol=1e-9, atol=0.0)


@pytest.mark.parametrize(("a", "b"), _ISF_SATURATING_PARAMS, ids=str)
def test_isf_saturates_to_one_only_where_no_float64_is_closer(a: float, b: float) -> None:
    """Where the answer rounds to the support edge, `isf` returns it, and never overshoots.

    The complement of a saturated answer is below `1.1e-16`, so `1 - isf(q)` cannot be represented:
    `1.0` is the correctly-rounded result and scipy returns it too. What this pins is that saturation
    is *monotone* and confined to the deep end, rather than the old failure where a rounded
    complement pulled the answer back to a wrong interior point.
    """
    dist = Beta(a=a, b=b)
    got = pl.DataFrame({"q": _ISF_DEEP_QUANTILES}).select(r=dist.isf(pl.col("q")))["r"].to_numpy()
    assert ((got >= 0.0) & (got <= 1.0)).all()
    assert (np.diff(got) <= 0.0).all(), "isf is not non-increasing in q"
    resolved = got < 1.0
    scipy_values = np.asarray(scipy_beta.isf(np.array(_ISF_DEEP_QUANTILES), a, b), dtype=float)
    np.testing.assert_allclose(got[resolved], scipy_values[resolved], rtol=1e-8, atol=0.0)


@pytest.mark.parametrize("q", [1e-9, 1.0695737053549547e-09, 1e-4, 0.25])
def test_isf_matches_scipy_where_scipy_still_resolves(q: float) -> None:
    """Above `q ~ 1e-8` scipy is a valid oracle, and the fix must not have moved anything there.

    The round-trip test above cannot catch a systematic offset shared by `isf` and `log_sf`; this
    one anchors the absolute scale against an independent implementation, in the only regime where
    that implementation is trustworthy.
    """
    for a, b in _ISF_RESOLVING_PARAMS:
        got = pl.DataFrame({"q": [q]}).select(r=Beta(a=a, b=b).isf(pl.col("q")))["r"].item()
        assert got == pytest.approx(float(scipy_beta.isf(q, a, b)), rel=1e-8)


@pytest.mark.parametrize(("a", "b"), [(0.05, 0.05), (0.5, 0.5), (0.1, 500.0)])
def test_density_diverges_at_a_boundary_whose_shape_is_below_one(a: float, b: float) -> None:
    """`pdf` and `log_pdf` agree with scipy, and with each other, where the density diverges.

    `statrs` disagreed with itself: `ln_pdf` was `-inf` at both endpoints unconditionally, while
    `pdf` was `inf` for small shapes but `0.0` once it switched to `ln_pdf().exp()` above shape ~80.
    scipy returns `inf` for both, which is the limit.
    """
    edges = [0.0, 1.0]
    got = pl.DataFrame({"x": edges}).select(
        pdf=Beta(a=a, b=b).pdf(pl.col("x")),
        log_pdf=Beta(a=a, b=b).log_pdf(pl.col("x")),
    )
    np.testing.assert_array_equal(got["pdf"].to_numpy(), scipy_beta.pdf(edges, a, b))
    np.testing.assert_array_equal(got["log_pdf"].to_numpy(), scipy_beta.logpdf(edges, a, b))
