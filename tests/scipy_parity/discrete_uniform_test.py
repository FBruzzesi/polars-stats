from __future__ import annotations

import polars as pl
import pytest
from scipy.stats import randint as scipy_randint

from polars_stats import DiscreteUniform
from tests.scipy_parity._harness import Case, assert_case_matches_scipy

# Parameter and evaluation grids for the parity sweep. Owned by this test category and independent
# of the per-method functional tests under `tests/distributions/discrete_uniform`.
# Both bounds are INCLUSIVE here; scipy's `high` is exclusive, so every frozen reference is built
# with `high=max + 1`. That reparameterisation is the loudest in the catalogue and is itself under
# test: a swapped convention would fail every row below.
_BOUNDS = [(0, 5), (-4, 4), (1, 2), (7, 60), (-20, -10)]
# Interior quantiles only, for two reasons: scipy's discrete ppf/isf return the below-support
# sentinel `low - 1` at the exact endpoints q in {0, 1}, which this support-clamped inverse does
# not reproduce (same divergence as every other family); and the one-point mass `min == max` is
# excluded from this sweep entirely because scipy's own `_stats` divides by zero there.
_QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
# Spans below the support, integer points on it, non-integers, and past the inclusive max.
_VALUE_GRIDS = {
    # A per-bound grid keeps the off-support probes near each support.
    (0, 5): [-3.0, -1.0, 0.5, 1.0, 2.0, 3.7, 5.0, 8.0],
    (-4, 4): [-9.0, -4.0, -1.5, 0.0, 2.0, 4.0, 6.0],
    (1, 2): [0.0, 0.5, 1.0, 1.6, 2.0, 3.0],
    (7, 60): [2.0, 7.0, 15.0, 30.5, 44.0, 60.0, 61.0],
    (-20, -10): [-25.0, -20.0, -14.0, -11.0, -10.0, -5.0],
}


def _value_grid(bounds: tuple[int, int]) -> list[float]:
    return _VALUE_GRIDS[bounds]


_CASES: list[Case[DiscreteUniform]] = [
    Case("pmf", "value", lambda d, c: d.pmf(c), "pmf"),
    Case("log_pmf", "value", lambda d, c: d.log_pmf(c), "logpmf"),
    Case("cdf", "value", lambda d, c: d.cdf(c), "cdf"),
    Case("log_cdf", "value", lambda d, c: d.log_cdf(c), "logcdf"),
    Case("sf", "value", lambda d, c: d.sf(c), "sf"),
    Case("log_sf", "value", lambda d, c: d.log_sf(c), "logsf"),
    Case("ppf", "quantile", lambda d, c: d.ppf(c), "ppf"),
    Case("isf", "quantile", lambda d, c: d.isf(c), "isf"),
    Case("mean", "scalar", lambda d, _: d.mean(), "mean"),
    Case("variance", "scalar", lambda d, _: d.variance(), "var"),
    Case("std", "scalar", lambda d, _: d.std(), "std"),
    Case("entropy", "scalar", lambda d, _: d.entropy(), "entropy"),
]


@pytest.mark.parametrize("bounds", _BOUNDS, ids=[f"bounds={b}" for b in _BOUNDS])
@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
def test_method_matches_scipy(case: Case[DiscreteUniform], bounds: tuple[int, int]) -> None:
    """Every closed-form method matches `scipy.stats.randint(low=min, high=max + 1)` across grids.

    Table-driven: adding a method is one row in `_CASES`. `median` is deliberately absent: this
    crate reports the midpoint `(min + max) / 2` per the issue's convention, where scipy answers
    `ppf(0.5)`, a support point for even-width supports -- a documented divergence asserted in
    `tests/distributions/discrete_uniform/median_test.py` instead. Method-specific behaviour
    (inclusive bounds, null propagation) is asserted in the per-method test files.
    """
    lo, hi = bounds
    assert_case_matches_scipy(
        case,
        dist=DiscreteUniform(min=lo, max=hi),
        scipy_frozen=scipy_randint(low=lo, high=hi + 1),
        value_grid=_value_grid(bounds),
        quantiles=_QUANTILES,
    )


def test_ppf_exact_integer_parity_across_a_full_support() -> None:
    """The closed-form ppf matches scipy to exact integer equality at every cdf step of a support.

    The issue requires verbatim parity including rounding at step boundaries; a shared closed form
    makes that hold even under half-ulp nudges of each step edge, which is where a one-sided
    rounding difference would show up as an off-by-one. The exact endpoints stay out of the grid:
    at q <= 0 / q >= 1 scipy answers its below-support sentinel (`low - 1`), this implementation
    clamps to the support -- the same documented divergence as every other family.
    """
    lo, hi = 1, 12
    n = hi - lo + 1
    edges = [k / n for k in range(1, n)]
    quantiles = sorted(set(edges + [e - 2**-53 for e in edges] + [e + 2**-53 for e in edges]))
    got = pl.DataFrame({"q": quantiles}).select(r=DiscreteUniform(min=lo, max=hi).ppf("q"))["r"]
    expected = [scipy_randint.ppf(q, low=lo, high=hi + 1) for q in quantiles]
    assert got.to_list() == expected


def test_cdf_max_is_one_where_an_exclusive_reading_is_not() -> None:
    """`cdf(max)` is exactly 1; treating the bound as exclusive, as scipy does, cannot say that.

    This is the behavioural marker of the inclusive convention: a port that passes `max` as
    `randint`'s exclusive `high` reads one support point fewer, answering
    `(max - min) / (max - min + 1)` at `max` instead of 1.
    """
    lo, hi = 1, 6
    ours = pl.DataFrame({"x": [float(hi)]}).select(r=DiscreteUniform(min=lo, max=hi).cdf("x"))["r"][0]
    assert ours == 1.0
    exclusive_reading = (hi - lo) / (hi - lo + 1)
    assert exclusive_reading < 1.0
