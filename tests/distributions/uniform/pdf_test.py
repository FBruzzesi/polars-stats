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
def test_pdf_matches_scipy(
    mn: float,
    mx: float,
    value_grid: Callable[[float, float], list[float]],
    scipy_uniform: Callable[[float, float], Any],
) -> None:
    xs = value_grid(mn, mx)
    got = pl.DataFrame({"x": xs}).select(r=Uniform(min=mn, max=mx).pdf(pl.col("x")))["r"].to_numpy()
    np.testing.assert_allclose(got, scipy_uniform(mn, mx).pdf(xs), atol=1e-12, rtol=0)


@pytest.mark.parametrize(("mn", "mx"), PARAMS)
def test_pdf_constant_inside_and_zero_outside(mn: float, mx: float) -> None:
    width = mx - mn
    inside = [mn + 0.1 * width, (mn + mx) / 2, mn + 0.9 * width]
    outside = [mn - width, mx + width]
    df = pl.DataFrame({"x": inside + outside})
    got = df.select(r=Uniform(min=mn, max=mx).pdf(pl.col("x")))["r"]
    expected = pl.Series("r", [1 / width] * len(inside) + [0.0] * len(outside), dtype=pl.Float64)
    assert_series_equal(got, expected)


def test_pdf_column_bounds() -> None:
    df = pl.DataFrame({"lo": [0.0, -2.0, 10.0], "hi": [1.0, 2.0, 20.0], "x": [0.5, 0.0, 25.0]})
    got = df.select(r=Uniform(min=pl.col("lo"), max=pl.col("hi")).pdf(pl.col("x")))["r"]
    expected = pl.Series("r", [1.0, 0.25, 0.0], dtype=pl.Float64)
    assert_series_equal(got, expected)


def test_pdf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"x": [0.5, None, 2.0]}, schema={"x": pl.Float64})
    got = df.select(r=Uniform(min=0.0, max=1.0).pdf(pl.col("x")))["r"]
    expected = pl.Series("r", [1.0, None, 0.0], dtype=pl.Float64)
    assert_series_equal(got, expected)
