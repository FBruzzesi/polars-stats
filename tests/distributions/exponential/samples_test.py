from __future__ import annotations

from typing import TYPE_CHECKING, cast

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import Exponential

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize("size", [1, 4, 16])
def test_samples_shape_and_dtype(size: int, frame: Callable[..., pl.DataFrame], seed: int) -> None:
    n = 32
    result = frame(size=n).select(s=Exponential(rate=1.0).samples(size=size, seed=seed))
    assert result.height == n
    assert result.schema["s"] == pl.Array(pl.Float64, size)


def test_samples_seed_reproducible(frame: Callable[..., pl.DataFrame], seed: int) -> None:
    dframe = frame(size=128)
    s1 = dframe.select(s=Exponential(rate=1.0).samples(size=8, seed=seed))["s"]
    s2 = dframe.select(s=Exponential(rate=1.0).samples(size=8, seed=seed))["s"]
    assert_series_equal(s1, s2)


def test_samples_columns_are_not_all_equal(frame: Callable[..., pl.DataFrame], seed: int) -> None:
    size = 8
    dframe = frame(size=512)
    result = dframe.select(s=Exponential(rate=1.0).samples(size=size, seed=seed))["s"]
    columns = [result.arr.get(i) for i in range(size)]
    distinct = {tuple(c.to_list()) for c in columns}
    assert len(distinct) == size


def test_samples_non_negative(frame: Callable[..., pl.DataFrame], seed: int) -> None:
    series = frame(size=2_000).select(s=Exponential(rate=2.0).samples(size=8, seed=seed))["s"].arr.explode()
    assert (series >= 0.0).all()


def test_samples_mean_close_to_inverse_rate(frame: Callable[..., pl.DataFrame], seed: int) -> None:
    series = frame(size=4_000).select(s=Exponential(rate=2.0).samples(size=16, seed=seed))["s"].arr.explode()
    mean = cast("float", series.mean())
    assert abs(mean - 0.5) < 0.03 * 0.5


@pytest.mark.parametrize("bad_size", [0, -1])
def test_samples_rejects_non_positive_size(bad_size: int) -> None:
    with pytest.raises(ValueError, match="size must be a positive integer"):
        Exponential(rate=1.0).samples(size=bad_size, seed=0)


def test_samples_null_rate_row_is_null_array(seed: int) -> None:
    size = 4
    dframe = pl.DataFrame({"rate": [1.0, None, 2.0]}, schema={"rate": pl.Float64})
    result = dframe.select(s=Exponential(rate=pl.col("rate")).samples(size=size, seed=seed))["s"]
    assert result.dtype == pl.Array(pl.Float64, size)
    # A null rate row yields a null array, not an array of inner-null elements.
    assert_series_equal(result.is_null(), pl.Series("s", [False, True, False]))


def test_samples_non_positive_rate_raises(seed: int) -> None:
    dframe = pl.DataFrame({"rate": [1.0, -0.5]})  # row 1: rate <= 0
    with pytest.raises(pl.exceptions.ComputeError, match="rate must be strictly positive"):
        dframe.select(s=Exponential(rate=pl.col("rate")).samples(size=4, seed=seed))
