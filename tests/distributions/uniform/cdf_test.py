from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import Uniform

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

PARAMS = [(0.0, 1.0), (-2.0, 3.0), (2.0, 5.0), (-5.0, -1.0), (0.0, 1e-3)]


@pytest.mark.parametrize(("mn", "mx"), PARAMS)
def test_cdf_matches_scipy(
    mn: float,
    mx: float,
    value_grid: Callable[[float, float], list[float]],
    scipy_uniform: Callable[[float, float], Any],
) -> None:
    xs = value_grid(mn, mx)
    got = pl.DataFrame({"x": xs}).select(r=Uniform(min=mn, max=mx).cdf(pl.col("x")))["r"].to_numpy()
    np.testing.assert_allclose(got, scipy_uniform(mn, mx).cdf(xs), atol=1e-12, rtol=0)


def test_cdf_clamps_outside_support() -> None:
    df = pl.DataFrame({"x": [-10.0, 0.0, 0.5, 1.0, 10.0]})
    got = df.select(r=Uniform(min=0.0, max=1.0).cdf(pl.col("x")))["r"]
    expected = pl.Series("r", [0.0, 0.0, 0.5, 1.0, 1.0], dtype=pl.Float64)
    assert_series_equal(got, expected)


def test_cdf_column_bounds() -> None:
    df = pl.DataFrame({"lo": [0.0, -2.0], "hi": [1.0, 2.0], "x": [0.5, 0.0]})
    got = df.select(r=Uniform(min=pl.col("lo"), max=pl.col("hi")).cdf(pl.col("x")))["r"]
    expected = pl.Series("r", [0.5, 0.5], dtype=pl.Float64)
    assert_series_equal(got, expected)


def test_cdf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"x": [0.25, None, 0.75]}, schema={"x": pl.Float64})
    got = df.select(r=Uniform(min=0.0, max=1.0).cdf(pl.col("x")))["r"]
    expected = pl.Series("r", [0.25, None, 0.75], dtype=pl.Float64)
    assert_series_equal(got, expected)
