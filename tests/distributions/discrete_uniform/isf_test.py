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
        (1, 6, 1.0, 1.0),  # every survival mass is <= 1; the smallest point wins
        (1, 6, 0.9, 1.0),
        (1, 6, 5 / 6, 1.0),  # sf(min) = 5/6 sits exactly on this step
        (1, 6, 0.8, 2.0),
        (1, 6, 0.5, 3.0),
        (1, 6, 1 / 6, 5.0),  # sf(5) = 1/6
        (1, 6, 1e-300, 6.0),  # only sf(max) = 0 satisfies it
        (1, 6, 0.0, 6.0),  # sf(max) = 0 already satisfies the inequality at the top point
        (-3, 2, 0.5, -1.0),  # sf(-1) = 1/2 sits exactly on this step
        (3, 3, 0.5, 3.0),
    ],
)
def test_isf_scalar(lo: int, hi: int, quantile: float, expected: float, unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=DiscreteUniform(min=lo, max=hi).isf(quantile)).item(0, "v")
    assert result == expected


@pytest.mark.parametrize("lo_hi", [(1, 6), (-5, 9), (0, 100)])
def test_isf_matches_scipy(lo_hi: tuple[int, int]) -> None:
    """Exact parity with `scipy.stats.randint.isf` at mid-step quantiles.

    Quantiles sit halfway between cdf steps, where every correct implementation has half a step of
    slack and must agree; exactly-at-step quantiles are decided by each implementation's own float
    path instead, and scipy's accumulates its cdf table by repeated addition so its edges drift.
    Endpoints are excluded too: scipy's generic inverse answers its below-support sentinel
    `low - 1` at exactly q = 1, where this implementation clamps to the support.
    """
    lo, hi = lo_hi[0], lo_hi[1]
    n = hi - lo + 1
    quantiles = [(k + 0.5) / n for k in range(n)]
    df = pl.DataFrame({"q": quantiles})
    got = df.select(r=DiscreteUniform(min=lo, max=hi).isf("q"))["r"]
    expected = [scipy_randint.isf(q, low=lo, high=hi + 1) for q in quantiles]
    assert got.to_list() == expected


@pytest.mark.parametrize("quantile", [-0.1, 1.5])
def test_isf_out_of_range_quantile_is_null(quantile: float, unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=DiscreteUniform(min=1, max=6).isf(quantile)).item(0, "v")
    assert result is None


def test_isf_propagates_nan_in_quantile(unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=DiscreteUniform(min=1, max=6).isf(float("nan"))).item(0, "v")
    assert math.isnan(result)


def test_isf_propagates_null_in_quantile() -> None:
    df = pl.DataFrame({"q": [0.1, None, 0.9]}, schema={"q": pl.Float64})
    result = df.select(v=DiscreteUniform(min=1, max=6).isf(pl.col("q")))["v"]
    assert_series_equal(result, pl.Series("v", [6.0, None, 1.0], dtype=pl.Float64))
