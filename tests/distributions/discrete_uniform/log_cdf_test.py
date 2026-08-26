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


def test_log_cdf_saturation_zone_reads_the_tail() -> None:
    # One support point short of the top, log_cdf must read log((N-1)/N), not the 0 a naive
    # `cdf().log()` gives once the cdf rounds to 1.
    lo, hi = 0, 10_000
    got = pl.DataFrame({"x": [float(hi - 1)]}).select(r=DiscreteUniform(min=lo, max=hi).log_cdf("x"))["r"][0]
    assert got == pytest.approx(math.log((hi - lo) / (hi - lo + 1)), rel=1e-12)


def test_log_cdf_propagates_null_in_bounds(bounds_with_null: pl.DataFrame) -> None:
    result = bounds_with_null.select(r=DiscreteUniform(min=pl.col("lo"), max=pl.col("hi")).log_cdf(2.0))["r"]
    assert result[0] == pytest.approx(math.log(2 / 6))
    assert result[1] is None
