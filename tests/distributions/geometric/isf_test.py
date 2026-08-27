from __future__ import annotations

import math

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import Geometric


@pytest.mark.parametrize(
    ("p", "quantile", "expected"),
    [
        (0.3, 1.0, 1.0),  # smallest support point, mapped explicitly at the degenerate endpoint
        (0.3, 0.71, 1.0),  # sf(1) = 0.7 <= 0.71
        (0.3, 0.69, 2.0),  # sf(1) = 0.7 > 0.69
        (0.3, 0.5, 2.0),
        (0.5, 0.5, 1.0),
        (0.5, 0.25, 2.0),
        (0.3, 0.0, float("inf")),
        (1.0, 0.5, 1.0),  # point mass at k = 1: every quantile inverts to it
        (1.0, 1e-300, 1.0),
        (1.0, 0.0, 1.0),  # sf(1) = 0 already satisfies the inequality at the mass point
    ],
)
def test_isf_scalar(p: float, quantile: float, expected: float, unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=Geometric(p=p).isf(quantile)).item(0, "v")
    assert result == expected


def test_isf_inverts_sf(unit_frame: pl.DataFrame) -> None:
    for p, q in [(0.1, 1e-4), (0.3, 1e-8), (0.7, 0.9)]:
        k = unit_frame.select(v=Geometric(p=p).isf(q)).item(0, "v")
        assert unit_frame.select(v=Geometric(p=p).sf(k)).item(0, "v") <= q
        assert unit_frame.select(v=Geometric(p=p).sf(k - 1)).item(0, "v") > q


def test_isf_deep_tail_keeps_full_precision(unit_frame: pl.DataFrame) -> None:
    # The base ppf(1 - q) quantises the tail mass to the spacing of 1.0 before the inverse
    # runs; entering against q itself must resolve q down past the underflow threshold.
    p, q = 0.5, 1e-300
    expected = math.ceil(math.log(q) / math.log1p(-p))
    got = unit_frame.select(v=Geometric(p=p).isf(q)).item(0, "v")
    assert got == expected


def test_isf_overshoots_by_one_at_an_exact_step_boundary(unit_frame: pl.DataFrame) -> None:
    """At `q = sf(1)` the answer is `2`, where `sf(1) <= q` already makes `1` the definition's answer.

    The direction is pinnable here, unlike on the `ppf` twin. `sf(1)` sits so close to `1` that the
    `log` back out of it is quantised to the spacing of `1.0`, which is `1e-8` *relative* at this
    `p`: the ratio misses the integer by `5e-9` rather than by a last bit, so every libm agrees on
    which side it falls. One ulp above the boundary the answer is `1` as expected, so the miss is
    one-sided in `q`, not a shifted support. Pinned, not fixed; see the reasoning on the `ppf` twin
    and in docs/explanation/accuracy.md, "Discrete `ppf` / `isf` at a step boundary".
    """
    p, support_floor, next_point = 1e-8, 1.0, 2.0
    sf_at_floor = unit_frame.select(v=Geometric(p=p).sf(support_floor)).item(0, "v")

    assert unit_frame.select(v=Geometric(p=p).isf(sf_at_floor)).item(0, "v") == next_point
    assert unit_frame.select(v=Geometric(p=p).isf(math.nextafter(sf_at_floor, 0.0))).item(0, "v") == next_point
    assert unit_frame.select(v=Geometric(p=p).isf(math.nextafter(sf_at_floor, 1.0))).item(0, "v") == support_floor


@pytest.mark.parametrize("quantile", [-0.1, 1.5])
def test_isf_out_of_range_quantile_is_null(quantile: float, unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=Geometric(p=0.3).isf(quantile)).item(0, "v")
    assert result is None


def test_isf_propagates_nan_in_quantile(unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=Geometric(p=0.3).isf(float("nan"))).item(0, "v")
    assert math.isnan(result)


def test_isf_propagates_null_in_quantile() -> None:
    df = pl.DataFrame({"q": [0.1, None, 0.9]}, schema={"q": pl.Float64})
    result = df.select(v=Geometric(p=0.3).isf(pl.col("q")))["v"]
    expected = pl.Series("v", [7.0, None, 1.0], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_isf_propagates_null_in_p(p_with_null: pl.DataFrame) -> None:
    result = p_with_null.select(v=Geometric(p=pl.col("p")).isf(0.5))["v"]
    expected = pl.Series("v", [2.0, None, 1.0], dtype=pl.Float64)
    assert_series_equal(result, expected)
