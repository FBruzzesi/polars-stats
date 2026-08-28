from __future__ import annotations

import math

import polars as pl
import pytest

from polars_stats import DiscreteUniform


@pytest.mark.parametrize(
    ("lo", "hi", "value", "expected"),
    [
        (1, 6, 0.5, float("-inf")),  # below the support: cdf == 0 exactly
        (1, 6, 1.0, math.log(1 / 6)),
        (1, 6, 6.0, 0.0),  # at the inclusive max the whole mass has accumulated
        (1, 6, 9.9, 0.0),
        (-3, 2, -4.5, float("-inf")),
    ],
)
def test_log_cdf_scalar(lo: int, hi: int, value: float, expected: float, unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=DiscreteUniform(min=lo, max=hi).log_cdf(value)).item(0, "v")
    assert result == expected


@pytest.mark.parametrize("n", [10_001, 10**5, 10**7, 10**9])
def test_log_cdf_saturation_zone_reads_the_tail(n: int) -> None:
    """One point below the top, `log_cdf` reads `log1p(-1/N)`, not `log` of a ratio rounded to 1."""
    lo, hi = 0, n - 1
    got = pl.DataFrame({"x": [float(hi - 1)]}).select(r=DiscreteUniform(min=lo, max=hi).log_cdf("x"))["r"][0]
    assert got == pytest.approx(math.log1p(-1.0 / n), rel=1e-15)


def test_log_cdf_propagates_null_in_bounds(bounds_with_null: pl.DataFrame) -> None:
    result = bounds_with_null.select(r=DiscreteUniform(min=pl.col("lo"), max=pl.col("hi")).log_cdf(2.0))["r"]
    assert result[0] == pytest.approx(math.log(2 / 6))
    assert result[1] is None
    assert result[2] is None


def test_log_cdf_saturates_to_zero_for_an_int_value_past_int64_half(unit_frame: pl.DataFrame) -> None:
    """`log_cdf` reads `0.0` from the `value >= max` branch for both the `int` and the `float` spelling."""
    d = DiscreteUniform(min=0, max=10)
    point = 2**63 - 1
    assert unit_frame.select(v=d.log_cdf(point)).item(0, "v") == 0.0
    assert unit_frame.select(v=d.log_cdf(float(point))).item(0, "v") == 0.0
