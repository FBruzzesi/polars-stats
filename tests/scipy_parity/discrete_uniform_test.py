from __future__ import annotations

import polars as pl
import pytest
from scipy.stats import randint as scipy_randint

from polars_stats import DiscreteUniform
from tests.scipy_parity._harness import Case, assert_case_matches_scipy

# Both bounds are INCLUSIVE here; scipy's `high` is exclusive, so every frozen reference is built
# with `high=max + 1`. A swapped convention would fail every row below.
_BOUNDS = [(0, 5), (-4, 4), (1, 2), (7, 60), (-20, -10)]
# Interior quantiles only: scipy answers its below-support sentinel `low - 1` at q in {0, 1}, and the
# one-point mass `min == max` is out of the sweep because scipy's own `_stats` divides by zero there.
_QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
# Below the support, integer points on it, non-integers, and past the inclusive max. Per-bound, so the
# off-support probes stay near each support.
_VALUE_GRIDS = {
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

    `median` is absent deliberately: this crate reports the midpoint where scipy answers `ppf(0.5)`,
    asserted in `tests/distributions/discrete_uniform/median_test.py` instead.
    """
    lo, hi = bounds
    assert_case_matches_scipy(
        case,
        dist=DiscreteUniform(min=lo, max=hi),
        scipy_frozen=scipy_randint(low=lo, high=hi + 1),
        value_grid=_value_grid(bounds),
        quantiles=_QUANTILES,
    )


@pytest.mark.parametrize(("lo", "hi"), [(1, 12), (1, 6), (-5, 9), (0, 100)])
def test_ppf_exact_integer_parity_across_a_full_support(lo: int, hi: int) -> None:
    """The closed-form ppf matches scipy to exact integer equality at every cdf step and both ulp neighbours.

    Integer equality rather than a float tolerance: a shared closed form should agree exactly, and a
    one-sided rounding difference shows up as an off-by-one. Endpoints stay out, per `_QUANTILES`.
    """
    n = hi - lo + 1
    edges = [k / n for k in range(1, n)]
    quantiles = sorted(set(edges + [e - 2**-53 for e in edges] + [e + 2**-53 for e in edges]))
    got = pl.DataFrame({"q": quantiles}).select(r=DiscreteUniform(min=lo, max=hi).ppf("q"))["r"]
    expected = [scipy_randint.ppf(q, low=lo, high=hi + 1) for q in quantiles]
    assert got.to_list() == expected


def test_cdf_max_is_one_where_an_exclusive_reading_is_not() -> None:
    """`cdf(max)` is exactly 1, which treating the bound as exclusive cannot say.

    The behavioural marker of the inclusive convention: passing `max` as `randint`'s exclusive `high`
    reads one support point fewer, answering `(max - min) / (max - min + 1)` at `max` instead of 1.
    """
    lo, hi = 1, 6
    ours = pl.DataFrame({"x": [float(hi)]}).select(r=DiscreteUniform(min=lo, max=hi).cdf("x"))["r"][0]
    assert ours == 1.0
    exclusive_reading = (hi - lo) / (hi - lo + 1)
    assert exclusive_reading < 1.0
