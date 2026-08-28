from __future__ import annotations

from fractions import Fraction

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import DiscreteUniform


@pytest.mark.parametrize(
    ("lo", "hi", "expected"),
    # For even N the midpoint falls between two support points; that is the answer, not a point.
    [(1, 6, 3.5), (-3, 2, -0.5), (3, 3, 3.0), (0, 7, 3.5)],
)
def test_mean(lo: int, hi: int, expected: float, unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=DiscreteUniform(min=lo, max=hi).mean()).item(0, "v")
    assert result == pytest.approx(expected)


def test_mean_propagates_null_in_bounds(bounds_with_null: pl.DataFrame) -> None:
    result = bounds_with_null.select(v=DiscreteUniform(min=pl.col("lo"), max=pl.col("hi")).mean())["v"]
    assert_series_equal(result, pl.Series("v", [3.5, None, None], dtype=pl.Float64))


@pytest.mark.parametrize(("lo", "hi"), [(2**62, 2**62 + 10), (2**62, 2**63 - 1)])
def test_mean_does_not_wrap_near_the_int64_ceiling(lo: int, hi: int, unit_frame: pl.DataFrame) -> None:
    """`min + max` overflows `Int64` well inside the range the validator accepts.

    Oracled by exact rational arithmetic rounded once, not the same sum spelled differently. A wrapped
    sum misses `[min, max]` by `9.2e18`, though containment is not a general invariant: for a support
    narrower than one float step the correctly rounded midpoint can sit half a step outside it.
    """
    result = unit_frame.select(v=DiscreteUniform(min=lo, max=hi).mean()).item(0, "v")
    assert result == float(Fraction(lo + hi, 2))
    assert lo <= result <= hi


@pytest.mark.parametrize(
    ("lo", "hi"),
    [
        (-4409513171799557951, 4409659924503471479),
        (-(2**61), 2**61 + 1),
        (-(2**62) + 3, 2**62 - 7),
    ],
)
def test_mean_is_exact_for_bounds_straddling_zero(lo: int, hi: int, unit_frame: pl.DataFrame) -> None:
    """Summing the bounds in `Float64` rounds both before they cancel; the `Int64` midpoint does not.

    The exact midpoint is representable for each of these pairs, so anything but equality is the
    cancellation defect rather than an unavoidable rounding.
    """
    exact = Fraction(lo + hi, 2)
    assert Fraction(float(exact)) == exact
    result = unit_frame.select(v=DiscreteUniform(min=lo, max=hi).mean()).item(0, "v")
    assert result == float(exact)
