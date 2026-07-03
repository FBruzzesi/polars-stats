from __future__ import annotations

import polars as pl
import pytest

from polars_stats import Beta
from tests._polars_compat import assert_series_equal


@pytest.mark.parametrize("shape", [0.5, 1.0, 2.0, 5.0])
def test_sf_is_half_at_centre_for_symmetric_shapes(shape: float) -> None:
    result = pl.DataFrame({"x": [0.5]}).select(r=Beta(a=shape, b=shape).sf(pl.col("x")))["r"].item()
    assert result == pytest.approx(0.5, abs=1e-12)


def test_sf_complements_cdf(params: tuple[float, float], value_grid: list[float]) -> None:
    a, b = params
    dist = Beta(a=a, b=b)
    result = pl.DataFrame({"x": value_grid}).select(total=dist.cdf(pl.col("x")) + dist.sf(pl.col("x")))["total"]
    expected = pl.Series("total", [1.0] * result.len(), dtype=pl.Float64)
    assert_series_equal(result, expected, rel_tol=0.0, abs_tol=1e-12)


def test_sf_clamps_outside_support() -> None:
    df = pl.DataFrame({"x": [-0.5, 0.0, 1.0, 1.5]})
    result = df.select(r=Beta(a=2.0, b=3.0).sf(pl.col("x")))["r"]
    expected = pl.Series("r", [1.0, 1.0, 0.0, 0.0], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_sf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"x": [0.5, None]}, schema={"x": pl.Float64})
    result = df.select(r=Beta(a=2.0, b=2.0).sf(pl.col("x")))["r"]
    expected = pl.Series("r", [0.5, None], dtype=pl.Float64)
    assert_series_equal(result, expected)
