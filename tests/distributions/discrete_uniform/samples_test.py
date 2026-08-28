from __future__ import annotations

from typing import TYPE_CHECKING, cast

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import DiscreteUniform
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
    result = frame(size=n).select(s=DiscreteUniform(min=1, max=6).samples(size=size, seed=seed))
    assert result.height == n
    assert result.schema["s"] == pl.Array(pl.Int64, size)


def test_samples_seed_reproducible(frame: Callable[..., pl.DataFrame], seed: int) -> None:
    dframe = frame(size=128)
    s1 = dframe.select(s=DiscreteUniform(min=-4, max=5).samples(size=8, seed=seed))["s"]
    s2 = dframe.select(s=DiscreteUniform(min=-4, max=5).samples(size=8, seed=seed))["s"]
    assert_series_equal(s1, s2)


def test_samples_columns_are_not_all_equal(
    frame: Callable[..., pl.DataFrame],
    seed: int,
) -> None:
    # Regression: the same seed must derive distinct sub-seeds so the `size`
    # columns are independent draws, not `size` copies of the same draw.
    size = 8
    dframe = frame(size=512)
    result = dframe.select(s=DiscreteUniform(min=1, max=50).samples(size=size, seed=seed))["s"]
    columns = [result.arr.get(i) for i in range(size)]
    distinct = {tuple(c.to_list()) for c in columns}
    assert len(distinct) == size


def test_samples_mean_close_to_midpoint_for_large_total(
    frame: Callable[..., pl.DataFrame],
    seed: int,
) -> None:
    n, size = 4_000, 16
    result = arr_explode(frame(n).select(s=DiscreteUniform(min=1, max=6).samples(size=size, seed=seed))["s"])
    mean = cast("float", result.mean())
    assert abs(mean - 3.5) < 0.01 * 5


def test_samples_scalar_fast_path_matches_per_row() -> None:
    n, size = 256, 4
    dframe = pl.DataFrame({"lo": [-3] * n, "hi": [7] * n}, schema={"lo": pl.Int64, "hi": pl.Int64})
    scalar = dframe.select(s=DiscreteUniform(min=-3, max=7).samples(size=size, seed=11))["s"]
    per_row = dframe.select(s=DiscreteUniform(min=pl.col("lo"), max=pl.col("hi")).samples(size=size, seed=11))["s"]
    assert_series_equal(scalar, per_row)


def test_samples_of_size_one_matches_sample(frame: Callable[..., pl.DataFrame], seed: int) -> None:
    # Both paths draw through the same `fn draw`, so the first draw of a row's stream is shared.
    dframe = frame(size=128)
    one = dframe.select(s=DiscreteUniform(min=1, max=6).samples(size=1, seed=seed))["s"].arr.get(0)
    single = dframe.select(k=DiscreteUniform(min=1, max=6).sample(seed=seed))["k"]
    assert_series_equal(one, single, check_names=False)


@pytest.mark.parametrize("bad_size", [0, -1])
def test_samples_rejects_non_positive_size(bad_size: int) -> None:
    with pytest.raises(ValueError, match="size must be a positive integer"):
        DiscreteUniform(min=1, max=6).samples(size=bad_size, seed=0)


def test_samples_null_bound_row_is_null_array(seed: int) -> None:
    # When a bound is null on a row, the entire array for that row must be null,
    # not an array of inner-null elements.
    size = 4
    dframe = pl.DataFrame({"lo": [1, None, 1], "hi": [6, 6, 6]}, schema={"lo": pl.Int64, "hi": pl.Int64})
    result = dframe.select(s=DiscreteUniform(min=pl.col("lo"), max=pl.col("hi")).samples(size=size, seed=seed))["s"]
    assert result.dtype == pl.Array(pl.Int64, size)
    assert_series_equal(result.is_null(), pl.Series("s", [False, True, False]))
