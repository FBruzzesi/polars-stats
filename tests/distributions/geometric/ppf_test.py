from __future__ import annotations

import math
from fractions import Fraction

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import Geometric


@pytest.mark.parametrize(
    ("p", "quantile", "expected"),
    [
        (0.3, 0.0, 1.0),  # smallest support point, mapped explicitly at the degenerate endpoint
        (0.3, 1e-300, 1.0),  # every representable q > 0 is >= cdf(1)
        (0.3, 0.29, 1.0),  # 0.29 <= cdf(1) = 0.3
        (0.3, 0.31, 2.0),  # boundary crossed between k=1 and k=2
        (0.3, 0.5, 2.0),
        (0.3, 0.51, 3.0),  # fl(0.51) > cdf(2) = 0.51: the stored quantile already crosses k = 2
        (0.5, 0.75, 2.0),  # exact power-of-two boundary: cdf(2) = 0.75 lands on the quantile bit-for-bit
        (0.3, 0.52, 3.0),
        (0.3, 1.0, float("inf")),  # unbounded support: only the limit answers q = 1
        (0.5, 0.5, 1.0),
        (1.0, 0.5, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
    ],
)
def test_ppf_scalar(p: float, quantile: float, expected: float, unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=Geometric(p=p).ppf(quantile)).item(0, "v")
    assert result == expected


def test_ppf_matches_closed_form_for_small_p(unit_frame: pl.DataFrame) -> None:
    # Interior-quantile check in a regime where a naive log(1 - q) / log(1 - p) loses
    # digits on both sides.
    p, q = 1e-13, 1e-11
    expected = math.ceil(math.log1p(-q) / math.log1p(-p))
    got = unit_frame.select(v=Geometric(p=p).ppf(q)).item(0, "v")
    assert got == expected


def _smallest_support_point_exact(p: float, quantile: float) -> int:
    """Smallest `k >= 1` with `cdf(k) >= quantile`, evaluated without rounding.

    `1 - (1 - p)**k >= q` over `Fraction`s, so this is the definition itself rather than a second
    floating-point implementation of it to compare against.
    """
    failure, target = 1 - Fraction(p), Fraction(quantile)
    k, tail = 1, failure
    while 1 - tail < target:
        k += 1
        tail *= failure
    return k


@pytest.mark.parametrize(("p", "step"), [(0.1, 10), (0.3, 4), (0.5, 20), (0.05, 13)])
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_ppf_at_an_exact_step_boundary_can_miss_by_one(
    p: float, step: int, offset: int, unit_frame: pl.DataFrame
) -> None:
    """On a step the answer is the definition's support point or a neighbour, and no further.

    The tie-break compares `k * log1p(-p)` against `log1p(-q)` rather than re-deriving `cdf(k)`.
    That is deliberate and the more accurate rule (1997 boundary probes: 357 disagreements with
    exact arithmetic against 490 for the alternative), and the price is that `ppf` and `cdf` are not
    exact mutual inverses on a step: the two roundings decide the last bit there. Which side they
    fall on is the platform's `exp` and `log1p`, not the rule, so only the bound is portable and
    only the bound is pinned. At `p = 0.1` one ulp above `cdf(10)`, Apple's libm answers `10` and
    glibc `11`, because their `exp` puts `cdf(10)` itself an ulp apart. Away from a step the answer
    is exact, as the scalar cases above and the `isf` twin, whose probe amplifies a round trip well
    past a last bit, pin. See docs/explanation/accuracy.md, "Discrete `ppf` / `isf` at a step
    boundary".
    """
    cdf_at_step = unit_frame.select(v=Geometric(p=p).cdf(float(step))).item(0, "v")
    quantile = cdf_at_step if offset == 0 else math.nextafter(cdf_at_step, float(offset > 0))

    got = unit_frame.select(v=Geometric(p=p).ppf(quantile)).item(0, "v")
    assert abs(got - _smallest_support_point_exact(p, quantile)) <= 1


@pytest.mark.parametrize("quantile", [-0.1, 1.5])
def test_ppf_out_of_range_quantile_is_null(quantile: float, unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=Geometric(p=0.3).ppf(quantile)).item(0, "v")
    assert result is None


def test_ppf_propagates_nan_in_quantile(unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=Geometric(p=0.3).ppf(float("nan"))).item(0, "v")
    assert math.isnan(result)


def test_ppf_propagates_null_in_quantile() -> None:
    df = pl.DataFrame({"q": [0.1, None, 0.9]}, schema={"q": pl.Float64})
    result = df.select(v=Geometric(p=0.3).ppf(pl.col("q")))["v"]
    expected = pl.Series("v", [1.0, None, 7.0], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_ppf_propagates_null_in_p(p_with_null: pl.DataFrame) -> None:
    result = p_with_null.select(v=Geometric(p=pl.col("p")).ppf(0.5))["v"]
    expected = pl.Series("v", [2.0, None, 1.0], dtype=pl.Float64)
    assert_series_equal(result, expected)
