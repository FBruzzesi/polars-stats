from __future__ import annotations

import math

import polars as pl
import pytest
from polars.testing import assert_series_equal

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
    # Integer equality, not the discrete tolerance band: the closed form matches scipy's own formula,
    # so there is no search slack to absorb.
    result = unit_frame.select(v=DiscreteUniform(min=lo, max=hi).ppf(quantile)).item(0, "v")
    assert result == expected


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


def test_ppf_of_a_constant_parameterisation_is_a_scalar_expression() -> None:
    """A scalar quantile against constant bounds is one value, not one per frame row."""
    df = pl.DataFrame({"g": [0, 0, 1, 1, 1]})
    grouped = df.group_by("g").agg(r=DiscreteUniform(min=0, max=5).ppf(0.5))
    assert grouped.schema["r"] == pl.Float64

    scalar = df.select(r=DiscreteUniform(min=0, max=5).ppf(0.5))
    assert scalar.height == 1
    assert_series_equal(scalar["r"], pl.Series("r", [2.0], dtype=pl.Float64))
