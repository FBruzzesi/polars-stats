from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
from scipy import special as scipy_special
from scipy.stats import beta as scipy_beta

from polars_stats import Beta
from tests.scipy_parity._harness import Case, assert_case_matches_scipy

# Parameter and evaluation grids for the parity sweep, owned by this test category. The grid covers
# the distinct shape regimes: unimodal, U-shaped (both shapes < 1), uniform (1, 1), J-shaped, skewed, and the
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
    """Reference `ln I_x(a, b)` where scipy's `logcdf` is already `-inf`.

    DLMF 8.17.7, `I_x(a, b) = x^a / (a B(a, b)) 2F1(a, 1 - b; a + 1; x)`, in logs. The `2F1` factor is `O(1)` and
    free of cancellation only for `b x <= 1`; at `(80, 90)`, `x = 0.4` it is catastrophically wrong, so every caller
    gates on that predicate. It holds at every point where scipy underflows, which needs `a ln x < ~-700` and so a
    tiny `x`.
    """
    xs_arr = np.asarray(xs)
    return (
        a * np.log(xs_arr)
        - math.log(a)
        - scipy_special.betaln(a, b)
        + np.log(scipy_special.hyp2f1(a, 1.0 - b, a + 1.0, xs_arr))
    )


# Each grid spans scipy's finite band and crosses its underflow point, the log prefactor below ~-745.
_LEFT_CORNER_CASES = [
    (200.0, 2.0, [1e-200, 1e-4, 0.01, 0.05, 0.3, 0.5]),
    (80.0, 90.0, [1e-200, 1e-4, 0.01, 0.05, 0.2, 0.4]),
    (2.0, 8.0, [1e-200, 1e-8, 1e-4, 0.01, 0.1]),
]


def _assert_scipy_or_series(got: np.ndarray, ref: np.ndarray, series: np.ndarray, series_safe: np.ndarray) -> None:
    finite = np.isfinite(ref)
    assert np.isfinite(got).all(), "a log tail underflowed to -inf"
    assert finite.any(), "the grid never reaches scipy's finite band"
    assert (~finite).any(), "the grid never crosses scipy's underflow point"
    assert (finite | series_safe).all(), "a point beyond scipy fell outside the series reference's domain"
    np.testing.assert_allclose(got[finite], ref[finite], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(got[series_safe], series, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize(("a", "b", "xs"), _LEFT_CORNER_CASES, ids=lambda v: str(v)[:16])
def test_log_cdf_left_corner_matches_scipy_or_series(a: float, b: float, xs: list[float]) -> None:
    """`log_cdf` stays finite deep in the left corner, where `cdf().log()` and scipy's `logcdf` are `-inf`."""
    got = pl.DataFrame({"x": xs}).select(r=Beta(a=a, b=b).log_cdf(pl.col("x")))["r"].to_numpy()
    series_safe = np.asarray([b * x <= 1.0 for x in xs])
    series = _series_ln_beta_cdf(a, b, [x for x in xs if b * x <= 1.0])
    _assert_scipy_or_series(got, scipy_beta(a=a, b=b).logcdf(xs), series, series_safe)


# The mirror corner, at `1 - y`: `sf(1 - y) = I_y(b, a)`. Underflow there needs a large `b` and `y` above ~1e-16,
# below which `1 - y` rounds to `1`, so these pairs are not mirrors of the left-corner ones.
_RIGHT_CORNER_CASES = [
    (2.0, 200.0, [1e-4, 0.01, 0.05, 0.3, 0.5]),
    (90.0, 80.0, [1e-16, 1e-4, 0.01, 0.05, 0.2, 0.4]),
    (8.0, 200.0, [1e-4, 0.01, 0.05, 0.1, 0.3]),
]


@pytest.mark.parametrize(("a", "b", "ys"), _RIGHT_CORNER_CASES, ids=lambda v: str(v)[:16])
def test_log_sf_right_corner_matches_scipy_or_series(a: float, b: float, ys: list[float]) -> None:
    """`log_sf` stays finite deep in the right corner, where `sf().log()` and scipy's `logsf` are `-inf`.

    The series reference is read at `1 - (1 - y)`, the complement the rounded input actually carries.
    """
    vs = [1.0 - y for y in ys]
    carried = [1.0 - v for v in vs]
    got = pl.DataFrame({"v": vs}).select(r=Beta(a=a, b=b).log_sf(pl.col("v")))["r"].to_numpy()
    series_safe = np.asarray([a * y <= 1.0 for y in carried])
    series = _series_ln_beta_cdf(b, a, [y for y in carried if a * y <= 1.0])
    _assert_scipy_or_series(got, scipy_beta(a=a, b=b).logsf(vs), series, series_safe)


# Where the opposite tail is small but far above underflow, scipy's linear complement is exact, so its `log1p` is
# the reference; scipy's own `logcdf` / `logsf` are the naive log of the rounded value there.
_NEAR_CERTAIN_CASES = [
    (2.0, 3.0, [0.9, 0.99, 1.0 - 1e-9], [1e-9, 1e-3, 0.05]),
    (90.0, 100.0, [0.6, 0.65, 0.7], [0.3, 0.35, 0.4]),
    (0.05, 0.05, [0.6, 0.9, 1.0 - 1e-6], [1e-6, 0.1, 0.4]),
]


@pytest.mark.parametrize(("a", "b", "upper", "lower"), _NEAR_CERTAIN_CASES, ids=lambda v: str(v)[:16])
def test_log_tails_keep_relative_precision_near_zero(
    a: float, b: float, upper: list[float], lower: list[float]
) -> None:
    """`log_cdf` near `1` and `log_sf` near `0` keep their relative precision."""
    frozen = scipy_beta(a=a, b=b)
    got = pl.DataFrame({"hi": upper, "lo": lower}).select(
        log_cdf=Beta(a=a, b=b).log_cdf(pl.col("hi")), log_sf=Beta(a=a, b=b).log_sf(pl.col("lo"))
    )
    np.testing.assert_allclose(got["log_cdf"].to_numpy(), np.log1p(-frozen.sf(upper)), rtol=1e-11, atol=0.0)
    np.testing.assert_allclose(got["log_sf"].to_numpy(), np.log1p(-frozen.cdf(lower)), rtol=1e-11, atol=0.0)


_POWER_SHAPES = [1e-12, 1e-8, 0.3, 1.0, 3.0, 1e3, 1e7]
_POWER_POINTS = [1e-200, 1e-9, 0.25, 0.5, 0.999999]


@pytest.mark.parametrize("shape", _POWER_SHAPES, ids=lambda s: f"s={s:g}")
def test_log_tails_match_the_power_law_closed_forms(shape: float) -> None:
    """`Beta(s, 1)` has `cdf = x^s` and `Beta(1, s)` has `sf = (1 - x)^s`, exact references at every shape.

    At a shape near `0` the tail sits next to `1` and its complement, `-expm1(s ln x)`, carries the answer: that is
    the power-series branch, whose exponent must be relatively exact as the shape vanishes. `1e-13`, not `1e-14`: a
    complement read back from a tiny tail's logarithm, `-exp(ln 1e-200)`, carries that log's rounding, `460 eps`.
    """
    frame = pl.DataFrame({"x": _POWER_POINTS})
    got = frame.select(
        cdf=Beta(a=shape, b=1.0).log_cdf(pl.col("x")),
        sf_of_cdf=Beta(a=shape, b=1.0).log_sf(pl.col("x")),
        sf=Beta(a=1.0, b=shape).log_sf(pl.col("x")),
    )
    ln_cdf = [shape * math.log(x) for x in _POWER_POINTS]
    ln_sf = [shape * math.log1p(-x) for x in _POWER_POINTS]
    np.testing.assert_allclose(got["cdf"].to_numpy(), ln_cdf, rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(got["sf"].to_numpy(), ln_sf, rtol=1e-13, atol=0.0)
    complement = [math.log1p(-math.exp(t)) if t < -math.log(2.0) else math.log(-math.expm1(t)) for t in ln_cdf]
    np.testing.assert_allclose(got["sf_of_cdf"].to_numpy(), complement, rtol=1e-13, atol=0.0)


def _ln_sf_integer_b(a: float, b: int, x: float) -> float:
    """`ln(1 - I_x(a, b)) = ln I_y(b, a)` for `a < 1` and an integer `b`, by DLMF 8.17.7 on the reflected side:
    `I_y(b, a) = y^b / B(b, a) sum_k (1 - a)_k y^k / (k! (b + k))`, `1 / B(b, a) = a prod_{0<i<b} (a + i) / (b - 1)!`.

    Every term is positive, so plain floats hold it to a few ulp; the textbook `1 - x^a sum_{j < b} ...` cancels to
    `O(a)` and does not.
    """
    y = 1.0 - x
    coefficient, series, k = 1.0, 1.0 / b, 0
    while True:
        k += 1
        coefficient *= (k - a) * y / k
        term = coefficient / (b + k)
        series += term
        if term < 1e-17 * series:
            break
    ln_inverse_beta = math.log(a) + sum(math.log(a + i) for i in range(1, b)) - math.lgamma(b)
    return b * math.log(y) + ln_inverse_beta + math.log(series)


@pytest.mark.parametrize("b", [2, 5, 30])
@pytest.mark.parametrize("a", [1e-12, 1e-8, 1e-4])
def test_log_sf_keeps_a_vanishing_first_shape(a: float, b: int) -> None:
    """`log_sf` at a first shape near `0`, where `I_x` is within `O(a)` of `1` and the complement carries the answer.

    The series exponent `ln Gamma(a + b) - ln Gamma(b) - ln Gamma(1 + a)` is `O(a)`; read as a difference of `ln_gamma`
    values, or of two Stirling remainders near `1 / 120`, it keeps `1e-18` absolute and loses the complement's digits.
    """
    xs = [0.01, 0.3, 0.6]
    got = pl.DataFrame({"x": xs}).select(r=Beta(a=a, b=float(b)).log_sf(pl.col("x")))["r"].to_numpy()
    np.testing.assert_allclose(got, [_ln_sf_integer_b(a, b, x) for x in xs], rtol=1e-14, atol=0.0)


@pytest.mark.parametrize("shape", [1e2, 1e4, 1e6], ids=lambda s: f"s={s:.0e}")
def test_log_tails_hold_the_symmetry_identity_at_large_shapes(shape: float) -> None:
    """`I_0.5(s, s) = 1/2` at every shape, so both log tails are `-ln 2` there.

    `ln_gamma(2 s) - 2 ln_gamma(s)` cancels to `~eps ln_gamma(2 s)` absolute, `2.8e-9` at `s = 1e6`; the Stirling
    form of the prefactor has nothing to cancel. A 141-term continued fraction, statrs' cap, answered `+0.76`.
    """
    dist = Beta(a=shape, b=shape)
    got = pl.DataFrame({"x": [0.5]}).select(lc=dist.log_cdf(pl.col("x")), ls=dist.log_sf(pl.col("x"))).row(0)
    assert got == pytest.approx((-math.log(2.0), -math.log(2.0)), rel=0.0, abs=1e-12)


def test_log_tails_past_the_term_cap_are_nan() -> None:
    """At the mean of `a = b = 1e17` the continued fraction runs out of terms: `NaN`, never a truncated value."""
    dist = Beta(a=1e17, b=1e17)
    got = pl.DataFrame({"x": [0.5]}).select(lc=dist.log_cdf(pl.col("x")), ls=dist.log_sf(pl.col("x"))).row(0)
    assert all(math.isnan(value) for value in got)
