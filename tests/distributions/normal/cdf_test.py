from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import Normal

if TYPE_CHECKING:
    from collections.abc import Callable


def test_cdf_is_half_at_mean(params: tuple[float, float]) -> None:
    mean, std = params
    result = pl.DataFrame({"x": [mean]}).select(r=Normal(mean=mean, std_dev=std).cdf(pl.col("x")))["r"].item()
    assert result == pytest.approx(0.5, abs=1e-12)


def test_cdf_is_monotone_non_decreasing(
    params: tuple[float, float], value_grid: Callable[[float, float], list[float]]
) -> None:
    mean, std = params
    xs = value_grid(mean, std)
    result = pl.DataFrame({"x": xs}).select(r=Normal(mean=mean, std_dev=std).cdf(pl.col("x")))["r"]
    assert result.is_sorted()
    assert result.is_between(0.0, 1.0).all()


def test_cdf_column_params() -> None:
    df = pl.DataFrame({"mu": [0.0, 5.0], "sigma": [1.0, 2.0], "x": [0.0, 5.0]})
    result = df.select(r=Normal(mean=pl.col("mu"), std_dev=pl.col("sigma")).cdf(pl.col("x")))["r"]
    expected = pl.Series("r", [0.5, 0.5], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_cdf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"x": [0.0, None]}, schema={"x": pl.Float64})
    result = df.select(r=Normal().cdf(pl.col("x")))["r"]
    expected = pl.Series("r", [0.5, None], dtype=pl.Float64)
    assert_series_equal(result, expected)
