from __future__ import annotations

import math
from fractions import Fraction

import polars as pl
import pytest
from scipy.stats import uniform as scipy_uniform

from polars_stats import Uniform
from tests.scipy_parity._harness import Case, assert_case_matches_scipy

# Parameter and evaluation grids for the parity sweep. Owned by this test category and independent
# of the per-method functional tests under `tests/distributions/uniform`.
_PARAMS = [(0.0, 1.0), (-2.0, 3.0), (2.0, 5.0), (-5.0, -1.0), (0.0, 1e-3)]
_QUANTILES = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]


_CASES: list[Case[Uniform]] = [
    Case("pdf", "value", lambda u, c: u.pdf(c), "pdf"),
    Case("log_pdf", "value", lambda u, c: u.log_pdf(c), "logpdf"),
    Case("cdf", "value", lambda u, c: u.cdf(c), "cdf"),
    Case("log_cdf", "value", lambda u, c: u.log_cdf(c), "logcdf"),
    Case("sf", "value", lambda u, c: u.sf(c), "sf"),
    Case("log_sf", "value", lambda u, c: u.log_sf(c), "logsf"),
    Case("ppf", "quantile", lambda u, c: u.ppf(c), "ppf"),
    Case("isf", "quantile", lambda u, c: u.isf(c), "isf"),
    Case("mean", "scalar", lambda u, _: u.mean(), "mean"),
    Case("variance", "scalar", lambda u, _: u.variance(), "var"),
    Case("std", "scalar", lambda u, _: u.std(), "std"),
    Case("median", "scalar", lambda u, _: u.median(), "median"),
    Case("entropy", "scalar", lambda u, _: u.entropy(), "entropy"),
]


def _value_grid(mn: float, mx: float) -> list[float]:
    """Evaluation points for a `(min, max)` support: below, both endpoints, interior, above."""
    width = mx - mn
    return [mn - width, mn, mn + 0.25 * width, (mn + mx) / 2, mn + 0.75 * width, mx, mx + width]


@pytest.mark.parametrize(("mn", "mx"), _PARAMS, ids=[f"min={mn},max={mx}" for mn, mx in _PARAMS])
@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
def test_method_matches_scipy(case: Case[Uniform], mn: float, mx: float) -> None:
    """Every closed-form method matches `scipy.stats.uniform` across the `(min, max)` grid.

    Table-driven: adding a method is one row in `_CASES`. Method-specific behaviour (support
    clamping, null propagation, out-of-range quantiles) is asserted in the per-method test files.
    """
    assert_case_matches_scipy(
        case,
        dist=Uniform(min=mn, max=mx),
        scipy_frozen=scipy_uniform(loc=mn, scale=mx - mn),
        value_grid=_value_grid(mn, mx),
        quantiles=_QUANTILES,
    )


# Regressions from `make audit`: the near-certain side of the log methods, which `_QUANTILES` and
# the parity grid never approach closely enough to expose.
_SPANS = [(0.0, 1.0), (-3.0, 7.0), (-1e8, 1e8), (0.0, 1e-8)]
_OFFSETS = [1e-16, 1e-12, 1e-8, 1e-4]


@pytest.mark.parametrize(("lo", "hi"), _SPANS, ids=str)
@pytest.mark.parametrize("offset", _OFFSETS, ids=lambda o: f"offset={o}")
def test_log_methods_keep_the_near_certain_side(lo: float, hi: float, offset: float) -> None:
    """`log_cdf` just below `max` and `log_sf` just above `min` keep relative precision.

    Both inherited a plain `log` of a ratio that rounds to exactly `1` as the argument approaches
    the far support edge, so the result collapsed to `0.0` where the truth is a small negative
    (`-8.9e-17` one ulp below `max` on `[-3, 7]`). The fix is the `log1p` branch `normal.rs`'s
    `ln_half_erfc` already takes; the oracle is `log1p(-offset)`, exact because the ratio is the
    fraction of the span and `offset` is it exactly.
    """
    width = hi - lo
    frame = pl.DataFrame({"upper": [hi - width * offset], "lower": [lo + width * offset]})
    got = frame.select(
        log_cdf=Uniform(min=lo, max=hi).log_cdf(pl.col("upper")),
        log_sf=Uniform(min=lo, max=hi).log_sf(pl.col("lower")),
    )
    expected = math.log1p(-offset)
    assert got["log_cdf"].item() == pytest.approx(expected, rel=1e-11)
    assert got["log_sf"].item() == pytest.approx(expected, rel=1e-11)


# The `isf` regression. `isf` was the base-class `ppf(1 - quantile)`, which forms
# `min + (1 - quantile) * range`; below `quantile ~ 1.1e-16` the complement rounds to exactly `1.0`
# and the whole tail collapses onto `min + range`. Invisible wherever `max` is far from zero, total
# wherever it is not. `_isf` now subtracts from `max`.
#
# Oracled by the exact closed form `max - quantile * range` in `Fraction` arithmetic, not by scipy:
# `scipy.stats.uniform.isf` composes the same way and reproduces the defect.
_ISF_SPANS = [(-1.0, 0.0), (0.0, 1.0), (-1.0, 1.0), (-1e10, 1.0), (-3.0, 7.0), (1e6, 1e6 + 1.0)]
_ISF_QUANTILES = [1e-300, 1e-100, 1e-20, 1e-17, 1e-16, 1e-9, 0.3, 0.5, 0.9]


@pytest.mark.parametrize(("lo", "hi"), _ISF_SPANS, ids=str)
def test_isf_keeps_relative_precision_where_the_answer_approaches_zero(lo: float, hi: float) -> None:
    """`isf` holds `1e-12` relative across the quantile range, on spans whose upper end is near zero.

    `Uniform(-1, 0).isf(1e-17)` returned `0.0` against a true `-1e-17` (100% relative error), and
    `Uniform(-1e10, 1).isf(1e-16)` was wrong by `9.1e-07`. Neither span was in the audit sweep, which
    is why a defect this large sat in a distribution whose every method is one line of arithmetic.
    """
    got = pl.DataFrame({"q": _ISF_QUANTILES}).select(r=Uniform(min=lo, max=hi).isf(pl.col("q")))["r"].to_list()
    span = Fraction(hi) - Fraction(lo)
    for q, value in zip(_ISF_QUANTILES, got, strict=True):
        exact = Fraction(hi) - Fraction(q) * span
        assert value == pytest.approx(float(exact), rel=1e-12, abs=0.0), f"isf({q}) on ({lo}, {hi})"


# The `ppf` half of the same reassociation, which shipped without a test while its `isf` twin above
# got one. `min + quantile * range` is a difference of nearly equal numbers once the result lands
# near `max`, so `Uniform(-1e10, 1).ppf(1 - 1.1e-10)` was relatively wrong by `3.0e-06`; `_ppf` now
# interpolates from `max` above the median. Every quantile below is above `0.5`, the only side the
# rewrite changed, and the spans are wide enough that `min` and the answer differ by many decades.
#
# Same `Fraction` oracle as the `isf` test and for the same reason: `scipy.stats.uniform.ppf` is
# `loc + q * scale`, the spelling under test, so it agrees with the defect.
_PPF_SPANS = [(-1e10, 1.0), (-1e10, 1e-10), (-1.0, 0.0), (0.0, 1.0), (-3.0, 7.0), (1e6, 1e6 + 1.0)]
_PPF_QUANTILES = [0.5, 0.9, 1 - 1.1e-10, 1 - 1e-12, 1 - 1e-15, 1 - 1.1e-16]


@pytest.mark.parametrize(("lo", "hi"), _PPF_SPANS, ids=str)
def test_ppf_keeps_relative_precision_near_the_upper_bound(lo: float, hi: float) -> None:
    """`ppf` holds `1e-12` relative just below `quantile = 1`, on spans far wider than the answer.

    `Uniform(-1e10, 1).ppf(1 - 1.1e-10)` was relatively wrong by `3.0e-06`. The parity grid's widest
    span is `(-1e8, 1e8)` and its quantiles stop at `0.999`, which is why arithmetic this simple
    could lose six digits with the whole suite green.
    """
    got = pl.DataFrame({"q": _PPF_QUANTILES}).select(r=Uniform(min=lo, max=hi).ppf(pl.col("q")))["r"].to_list()
    span = Fraction(hi) - Fraction(lo)
    for q, value in zip(_PPF_QUANTILES, got, strict=True):
        exact = Fraction(lo) + Fraction(q) * span
        assert value == pytest.approx(float(exact), rel=1e-12, abs=0.0), f"ppf({q}) on ({lo}, {hi})"
