from __future__ import annotations

import math

import polars as pl
import pytest
from polars.testing import assert_series_equal
from scipy.stats import randint as scipy_randint

from polars_stats import DiscreteUniform


@pytest.mark.parametrize(
    ("lo", "hi", "quantile", "expected"),
    [
        (1, 6, 0.0, 1.0),  # smallest support point, lifted from the formula's `min - 1` by the clamp
        (1, 6, 1e-300, 1.0),  # any representable q > 0 is >= cdf(min)
        (1, 6, 1 / 6, 1.0),  # exactly on the first cdf step
        (1, 6, 0.2, 2.0),
        (1, 6, 0.5, 3.0),
        (1, 6, 1 / 6 + 1e-12, 2.0),  # just past a step
        (1, 6, 5 / 6, 5.0),
        (1, 6, 1.0 - 1e-300, 6.0),
        (1, 6, 1.0, 6.0),  # the inclusive max answers q = 1
        (-3, 2, 0.5, -1.0),  # cdf(-1) = 3/6 sits exactly on this step
        (3, 3, 0.5, 3.0),  # the point mass inverts to itself
        (3, 3, 0.0, 3.0),
    ],
)
def test_ppf_scalar(lo: int, hi: int, quantile: float, expected: float, unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=DiscreteUniform(min=lo, max=hi).ppf(quantile)).item(0, "v")
    # Held to integer equality rather than the discrete tolerance band: the closed form matches
    # scipy's own formula, so there is no search slack to absorb.
    assert result == expected


@pytest.mark.parametrize("lo_hi", [(1, 6), (-5, 9), (0, 100)])
def test_ppf_matches_scipy_at_step_boundaries(lo_hi: tuple[int, int]) -> None:
    """Exact parity with `scipy.stats.randint.ppf`, which shares this closed form.

    The grid walks every interior cdf step edge of a support plus half-ulp perturbations of them,
    where a one-sided rounding difference would show up as an off-by-one; scipy resolves those
    through the same candidate-correction this ppf replicates. The exact endpoints stay out: at
    q <= 0 / q >= 1 scipy answers its below-support sentinel `low - 1`, this implementation clamps
    to the support.
    """
    lo, hi = lo_hi[0], lo_hi[1]
    n = hi - lo + 1
    edges = [k / n for k in range(1, n)]
    quantiles = sorted(set(edges + [e - 2**-53 for e in edges] + [e + 2**-53 for e in edges]))
    df = pl.DataFrame({"q": quantiles})
    got = df.select(r=DiscreteUniform(min=lo, max=hi).ppf("q"))["r"]
    expected = [scipy_randint.ppf(q, low=lo, high=hi + 1) for q in quantiles]
    assert got.to_list() == expected


@pytest.mark.parametrize("quantile", [-0.1, 1.5])
def test_ppf_out_of_range_quantile_is_null(quantile: float, unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=DiscreteUniform(min=1, max=6).ppf(quantile)).item(0, "v")
    assert result is None


def test_ppf_propagates_nan_in_quantile(unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=DiscreteUniform(min=1, max=6).ppf(float("nan"))).item(0, "v")
    assert math.isnan(result)


def test_ppf_propagates_null_in_quantile() -> None:
    df = pl.DataFrame({"q": [0.1, None, 0.9]}, schema={"q": pl.Float64})
    result = df.select(v=DiscreteUniform(min=1, max=6).ppf(pl.col("q")))["v"]
    assert_series_equal(result, pl.Series("v", [1.0, None, 6.0], dtype=pl.Float64))


def test_ppf_propagates_null_in_bounds(bounds_with_null: pl.DataFrame) -> None:
    result = bounds_with_null.select(v=DiscreteUniform(min=pl.col("lo"), max=pl.col("hi")).ppf(0.5))["v"]
    assert_series_equal(result, pl.Series("v", [3.0, None, None], dtype=pl.Float64))
