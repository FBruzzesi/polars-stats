from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import Beta

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize("size", [1, 4, 16])
def test_samples_shape_and_dtype(size: int, frame: Callable[..., pl.DataFrame], seed: int) -> None:
    n = 32
    result = frame(size=n).select(s=Beta(a=2.0, b=3.0).samples(size=size, seed=seed))
    assert result.height == n
    assert result.schema["s"] == pl.Array(pl.Float64, size)


def test_samples_seed_reproducible(frame: Callable[..., pl.DataFrame], seed: int) -> None:
    dframe = frame(size=128)
    s1 = dframe.select(s=Beta(a=2.0, b=3.0).samples(size=8, seed=seed))["s"]
    s2 = dframe.select(s=Beta(a=2.0, b=3.0).samples(size=8, seed=seed))["s"]
    assert_series_equal(s1, s2)


def test_samples_within_support(frame: Callable[..., pl.DataFrame], seed: int) -> None:
    result = frame(size=512).select(s=Beta(a=0.5, b=0.5).samples(size=8, seed=seed))["s"].arr.explode()
    assert result.is_between(0.0, 1.0).all()


def test_samples_columns_are_not_all_equal(frame: Callable[..., pl.DataFrame], seed: int) -> None:
    size = 8
    dframe = frame(size=512)
    result = dframe.select(s=Beta(a=2.0, b=3.0).samples(size=size, seed=seed))["s"]
    columns = [result.arr.get(i) for i in range(size)]
    distinct = {tuple(c.to_list()) for c in columns}
    assert len(distinct) == size


def test_samples_moments_close(frame: Callable[..., pl.DataFrame], seed: int) -> None:
    a, b = 2.0, 3.0
    mean = a / (a + b)
    std = math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
    result = frame(size=4_000).select(s=Beta(a=a, b=b).samples(size=16, seed=seed))["s"].arr.explode()
    _mean, _std = cast("float", result.mean()), cast("float", result.std())
    assert abs(_mean - mean) < 0.05 * std
    assert abs(_std - std) < 0.05 * std


@pytest.mark.parametrize("bad_size", [0, -1])
def test_samples_rejects_non_positive_size(bad_size: int) -> None:
    with pytest.raises(ValueError, match="size must be a positive integer"):
        Beta(a=2.0, b=3.0).samples(size=bad_size, seed=0)


def test_samples_null_param_row_is_null_array(seed: int) -> None:
    size = 4
    dframe = pl.DataFrame(
        {"a": [2.0, None, 1.0], "b": [3.0, 2.0, 1.0]},
        schema={"a": pl.Float64, "b": pl.Float64},
    )
    result = dframe.select(s=Beta(a=pl.col("a"), b=pl.col("b")).samples(size=size, seed=seed))["s"]
    assert result.dtype == pl.Array(pl.Float64, size)
    # A null param row yields a null array, not an array of inner-null elements.
    assert_series_equal(result.is_null(), pl.Series("s", [False, True, False]))


def test_samples_non_positive_shape_raises(seed: int) -> None:
    dframe = pl.DataFrame({"a": [2.0, 1.0], "b": [3.0, -2.0]})  # row 1: b = -2.0
    with pytest.raises(pl.exceptions.ComputeError, match="a and b must be finite and strictly positive"):
        dframe.select(s=Beta(a=pl.col("a"), b=pl.col("b")).samples(size=4, seed=seed))
