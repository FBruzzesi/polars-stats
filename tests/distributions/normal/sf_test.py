from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from polars_stats import Normal
from tests._polars_compat import assert_series_equal

if TYPE_CHECKING:
    from collections.abc import Callable


def test_sf_is_half_at_mean(params: tuple[float, float]) -> None:
    mean, std = params
    result = pl.DataFrame({"x": [mean]}).select(r=Normal(mu=mean, sigma=std).sf(pl.col("x")))["r"].item()
    assert result == pytest.approx(0.5, abs=1e-12)


def test_sf_complements_cdf(params: tuple[float, float], value_grid: Callable[[float, float], list[float]]) -> None:
    mean, std = params
    xs = value_grid(mean, std)
    n = Normal(mu=mean, sigma=std)
    result = pl.DataFrame({"x": xs}).select(total=n.cdf(pl.col("x")) + n.sf(pl.col("x")))["total"]
    expected = pl.Series("total", [1.0] * result.len(), dtype=pl.Float64)
    assert_series_equal(result, expected, rel_tol=0.0, abs_tol=1e-12)


def test_sf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"x": [0.0, None]}, schema={"x": pl.Float64})
    result = df.select(r=Normal().sf(pl.col("x")))["r"]
    expected = pl.Series("r", [0.5, None], dtype=pl.Float64)
    assert_series_equal(result, expected)
