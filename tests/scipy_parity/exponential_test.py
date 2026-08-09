from __future__ import annotations

import math

import polars as pl
import pytest
from scipy.stats import expon as scipy_expon

from polars_stats import Exponential
from tests.scipy_parity._harness import Case, assert_case_matches_scipy

# Parameter and evaluation grids for the parity sweep. Owned by this test category and independent
# of the per-method functional tests under `tests/distributions/exponential`.
_PARAMS = [0.1, 0.5, 1.0, 2.0, 5.0]
# Endpoints excluded: `ppf(1) = +inf` (the unbounded right tail) is asserted in `ppf_test.py`.
_QUANTILES = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]


# Exponential is elementary closed-form (no special functions), so every method agrees with scipy to
# machine precision and holds the harness default 1e-12. The `value_grid` stays on the open support
# `x > 0`: at `x <= 0` (`cdf = 0`) `log_cdf` is `-inf`, asserted in `log_cdf_test.py` instead.
_CASES: list[Case[Exponential]] = [
    Case("pdf", "value", lambda e, c: e.pdf(c), "pdf"),
    Case("log_pdf", "value", lambda e, c: e.log_pdf(c), "logpdf"),
    Case("cdf", "value", lambda e, c: e.cdf(c), "cdf"),
    Case("log_cdf", "value", lambda e, c: e.log_cdf(c), "logcdf"),
    Case("sf", "value", lambda e, c: e.sf(c), "sf"),
    Case("log_sf", "value", lambda e, c: e.log_sf(c), "logsf"),
    Case("ppf", "quantile", lambda e, c: e.ppf(c), "ppf"),
    Case("isf", "quantile", lambda e, c: e.isf(c), "isf"),
    Case("mean", "scalar", lambda e, _: e.mean(), "mean"),
    Case("variance", "scalar", lambda e, _: e.variance(), "var"),
    Case("std", "scalar", lambda e, _: e.std(), "std"),
    Case("median", "scalar", lambda e, _: e.median(), "median"),
    Case("entropy", "scalar", lambda e, _: e.entropy(), "entropy"),
]


def _value_grid(rate: float) -> list[float]:
    """Positive evaluation points, in multiples of the mean `1 / rate` (so `rate * x` is grid-fixed)."""
    mean = 1.0 / rate
    return [0.1 * mean, 0.25 * mean, 0.5 * mean, mean, 2.0 * mean, 4.0 * mean, 8.0 * mean]


@pytest.mark.parametrize("rate", _PARAMS, ids=[f"rate={r}" for r in _PARAMS])
@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
def test_method_matches_scipy(case: Case[Exponential], rate: float) -> None:
    """Every closed-form method matches `scipy.stats.expon(scale=1 / rate)` across the rate grid.

    Table-driven: adding a method is one row in `_CASES`. Method-specific behaviour (support edge,
    null propagation, out-of-range quantiles, the infinite ppf endpoint) is asserted in the
    per-method test files.
    """
    assert_case_matches_scipy(
        case,
        dist=Exponential(rate=rate),
        scipy_frozen=scipy_expon(scale=1.0 / rate),
        value_grid=_value_grid(rate),
        quantiles=_QUANTILES,
    )


# Regressions from `make audit`. The exponential is elementary throughout, so every oracle below is
# an exact limit rather than a reference implementation: `1 - exp(-t) -> t` and `-log1p(-q) -> q` as
# their arguments go to zero, both to full relative precision.

_RATES = [1e-8, 0.3, 1.0, 100.0, 1e8]
_TINY_ARGUMENTS = [1e-300, 1e-100, 1e-20, 1e-16, 1e-12]


@pytest.mark.parametrize("rate", _RATES, ids=lambda r: f"rate={r}")
@pytest.mark.parametrize("t", _TINY_ARGUMENTS, ids=lambda t: f"t={t}")
def test_cdf_keeps_the_left_tail_where_one_minus_exp_cancels(rate: float, t: float) -> None:
    """`cdf(t / rate)` is `t` to full relative precision, for `t` far below the `1 - exp` floor.

    Polars exposes no `expm1`, so `1 - exp(-t)` cancelled: below `t ~ 1.1e-16` the exponential
    rounds to exactly `1` and the cdf collapsed to `0`, while between there and `t ~ 1e-4` it
    carried a relative error of `~1.1e-16 / t` (11% at `1e-16`). `_cdf` now uses the identity
    `1 - exp(-t) = 2 exp(-t / 2) sinh(t / 2)`, which Polars can spell and which has no subtraction.
    scipy is a valid oracle only above its own cancellation floor, so the limit `cdf -> t` is used.
    """
    got = pl.DataFrame({"x": [t / rate]}).select(r=Exponential(rate=rate).cdf(pl.col("x")))["r"].item()
    assert got == pytest.approx(t, rel=1e-15)


@pytest.mark.parametrize("rate", _RATES, ids=lambda r: f"rate={r}")
@pytest.mark.parametrize("t", _TINY_ARGUMENTS, ids=lambda t: f"t={t}")
def test_log_cdf_is_finite_in_the_left_tail(rate: float, t: float) -> None:
    """`log_cdf` follows `cdf` into the left tail instead of saturating at `-inf`."""
    got = pl.DataFrame({"x": [t / rate]}).select(r=Exponential(rate=rate).log_cdf(pl.col("x")))["r"].item()
    assert got == pytest.approx(math.log(t), rel=1e-14)


@pytest.mark.parametrize("rate", _RATES, ids=lambda r: f"rate={r}")
@pytest.mark.parametrize("q", [1e-300, 1e-100, 1e-42, 1e-16, 1e-12, 0.25])
def test_ppf_keeps_tiny_quantiles(rate: float, q: float) -> None:
    """`ppf(q)` is `-log1p(-q) / rate`, exact for every `q` including ones `1 - q` cannot hold.

    Spelling it `-log(1 - q)` formed the complement first, so below `q ~ 1.1e-16` the quantile was
    thrown away and the result was `-0.0`; at `q = 1e-16` it was 11% wrong.
    """
    got = pl.DataFrame({"q": [q]}).select(r=Exponential(rate=rate).ppf(pl.col("q")))["r"].item()
    assert got == pytest.approx(-math.log1p(-q) / rate, rel=1e-15)


@pytest.mark.parametrize("rate", _RATES, ids=lambda r: f"rate={r}")
@pytest.mark.parametrize("q", [1e-300, 1e-100, 1e-20, 1e-9, 1e-3, 0.75])
def test_isf_is_exact_rather_than_ppf_of_the_complement(rate: float, q: float) -> None:
    """`isf(q)` is `-log(q) / rate`, never routed through `ppf(1 - q)`.

    The base-class default builds `1 - q`, whose absolute resolution is `1.1e-16`, so a small `q`
    was quantised (relative error `~1.1e-16 / q`, `1.4e-9` at `q = 1e-9`) and saturated below
    `1e-16`. The exponential has a closed-form inverse survival function, so it overrides.
    """
    got = pl.DataFrame({"q": [q]}).select(r=Exponential(rate=rate).isf(pl.col("q")))["r"].item()
    assert got == pytest.approx(-math.log(q) / rate, rel=1e-15)


# The `pdf` regression from `make audit`. `rate * exp(-rate * x)` rounds `exp` into the subnormal
# range and *then* scales that error up; `exp(log(rate) - rate * x)` rounds once, at the end.
#
# Oracled by hard-coded correctly-rounded values, verified against a 50-digit `mpmath` oracle at 0.13
# and 0.25 subnormal ulps respectively (against 42.9 and 4.3e7 before). scipy is not usable here: it
# forms the same product and reproduces the same wrong values.
_SUBNORMAL_PDF_CASES = [
    (100.0, 7.45, 2.8e-322),
    (1e8, 7.45e-06, 2.82235074e-316),
    (1e-8, 7.1e10, 4.476286e-317),
]


@pytest.mark.parametrize(("rate", "x", "expected"), _SUBNORMAL_PDF_CASES, ids=lambda v: str(v)[:10])
def test_pdf_is_correctly_rounded_in_the_subnormal_range(rate: float, x: float, expected: float) -> None:
    """`pdf` keeps every bit a subnormal has left, instead of scaling a truncated `exp` up."""
    got = pl.DataFrame({"x": [x]}).select(r=Exponential(rate=rate).pdf(pl.col("x")))["r"].item()
    assert got == expected


@pytest.mark.parametrize("rate", _PARAMS)
@pytest.mark.parametrize("x", [0.0, 1e-9, 0.5, 3.0, 50.0, 700.0])
def test_pdf_still_matches_scipy_above_the_subnormal_threshold(rate: float, x: float) -> None:
    """The log-space branch must not fire in the normal range, where it is the *less* accurate form.

    `exp(log(rate))` does not round-trip to `rate` for most rates, so computing everything in log
    space costs one to two orders of magnitude of relative accuracy everywhere the direct product is
    fine. The branch exists because of that, not despite it, and this pins that it stays on the right
    side of the threshold.
    """
    got = pl.DataFrame({"x": [x]}).select(r=Exponential(rate=rate).pdf(pl.col("x")))["r"].item()
    assert got == pytest.approx(float(scipy_expon.pdf(x, scale=1.0 / rate)), rel=1e-15, abs=0.0)
