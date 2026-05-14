from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import Bernoulli

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
    result = frame(n).select(s=Bernoulli(p=p).samples(size=size, seed=seed))["s"]
    flat = np.asarray(result.to_list(), dtype=float).ravel()
    assert abs(flat.mean() - p) < tolerance
