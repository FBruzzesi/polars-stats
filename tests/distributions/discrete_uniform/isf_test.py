from __future__ import annotations

import math
from fractions import Fraction

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


# Endpoints stay out: scipy's generic inverse answers its below-support sentinel `low - 1` at exactly
# q = 1, where this implementation clamps to the support.
@pytest.mark.parametrize(("lo", "hi"), [(1, 6), (-5, 9), (0, 100)])
def test_isf_matches_scipy(lo: int, hi: int) -> None:
    """Parity with `scipy.stats.randint.isf` at mid-step quantiles, where both have half a step of slack."""
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


# The grid displaces every step edge by half an ulp, where the rational answer is unambiguous and a
# bare `floor` of the rounded product loses a support point. The exactly representable edges are
# excluded deliberately: there `q` is the nearest float to `k / N` rather than `k / N` itself, so the
# rational contract and inverting this library's own `sf` disagree (see docs/explanation/accuracy.md).
@pytest.mark.parametrize(("lo", "hi"), [(1, 6), (0, 7), (-5, 9), (0, 100), (0, 999)])
def test_isf_matches_the_exact_closed_form_off_the_representable_edges(lo: int, hi: int) -> None:
    """Exact agreement with rational `max - floor(q * N)` on both sides of every step."""
    n = hi - lo + 1
    edges = [k / n for k in range(1, n)]
    quantiles = sorted({e - 2**-53 for e in edges} | {e + 2**-53 for e in edges})
    got = pl.DataFrame({"q": quantiles}).select(r=DiscreteUniform(min=lo, max=hi).isf("q"))["r"]
    expected = [float(min(max(hi - math.floor(Fraction(q) * n), lo), hi)) for q in quantiles]
    assert_series_equal(got, pl.Series("r", expected, dtype=pl.Float64))


@pytest.mark.parametrize(("lo", "hi"), [(1, 6), (0, 7), (-5, 9), (0, 100), (-20, -10), (3, 3), (0, 999)])
def test_isf_round_trips_every_support_point_to_within_one_step(lo: int, hi: int) -> None:
    """`isf(sf(x))` is `x` or the point above it, where `_sf`'s rounded `1 / N` and the probe's division disagree."""
    d = DiscreteUniform(min=lo, max=hi)
    points = pl.DataFrame({"x": [float(p) for p in range(lo, hi + 1)]})
    round_trip = points.select(x=pl.col("x"), rt=d.isf(d.sf("x")))
    assert round_trip.select(((pl.col("rt") == pl.col("x")) | (pl.col("rt") == pl.col("x") + 1)).all()).item()
    # The survival contract too; alone it is vacuous, since `sf(max) == 0` satisfies it for every `q`.
    quantiles = points.select(q=d.sf("x"))
    assert quantiles.select((d.sf(d.isf("q")) <= pl.col("q")).all()).item()
