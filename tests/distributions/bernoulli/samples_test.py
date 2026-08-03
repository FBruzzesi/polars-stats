from __future__ import annotations

from typing import TYPE_CHECKING, cast

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import Bernoulli
from tests._polars_compat import arr_explode

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize("size", [1, 4, 16])
def test_samples_shape_and_dtype(
    size: int,
    frame: Callable[..., pl.DataFrame],
    seed: int,
) -> None:
    n = 32
    result = frame(size=n).select(s=Bernoulli(p=0.5).samples(size=size, seed=seed))
    assert result.height == n
    assert result.schema["s"] == pl.Array(pl.Boolean, size)


def test_samples_seed_reproducible(frame: Callable[..., pl.DataFrame], seed: int) -> None:
    dframe = frame(size=128)
    s1 = dframe.select(s=Bernoulli(p=0.4).samples(size=8, seed=seed))["s"]
    s2 = dframe.select(s=Bernoulli(p=0.4).samples(size=8, seed=seed))["s"]
    assert_series_equal(s1, s2)


def test_samples_columns_are_not_all_equal(
    frame: Callable[..., pl.DataFrame],
    seed: int,
) -> None:
    # Regression: the same seed must derive distinct sub-seeds so the `size`
    # columns are independent draws, not `size` copies of the same draw.
    size = 8
    dframe = frame(size=512)
    result = dframe.select(s=Bernoulli(p=0.5).samples(size=size, seed=seed))["s"]
    columns = [result.arr.get(i) for i in range(size)]
    distinct = {tuple(c.to_list()) for c in columns}
    assert len(distinct) == size  # 8 truly independent draws → 8 distinct rasters


def test_samples_mean_close_to_p_for_large_total(
    frame: Callable[..., pl.DataFrame],
    seed: int,
) -> None:
    n, size, p = 4_000, 16, 0.3
    tolerance = 0.01
    result = arr_explode(frame(n).select(s=Bernoulli(p=p).samples(size=size, seed=seed))["s"])
    mean = cast("float", result.mean())
    assert abs(mean - p) < tolerance


@pytest.mark.parametrize("bad_size", [0, -1])
def test_samples_rejects_non_positive_size(bad_size: int) -> None:
    with pytest.raises(ValueError, match="size must be a positive integer"):
        Bernoulli(p=0.5).samples(size=bad_size, seed=0)


def test_samples_null_p_row_is_null_array(seed: int) -> None:
    # When `p` is null on a row, the entire array for that row must be null,
    # not an array of inner-null elements (which would still be a non-null array).
    size = 4
    dframe = pl.DataFrame({"p": [0.5, None, 0.5]}, schema={"p": pl.Float64})
    result = dframe.select(s=Bernoulli(p=pl.col("p")).samples(size=size, seed=seed))["s"]
    assert result.dtype == pl.Array(pl.Boolean, size)
    assert_series_equal(result.is_null(), pl.Series("s", [False, True, False]))
