from __future__ import annotations

from typing import TYPE_CHECKING, cast

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import LogNormal

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize("size", [1, 4, 16])
def test_samples_shape_and_dtype(size: int, frame: Callable[..., pl.DataFrame], seed: int) -> None:
    n = 32
    result = frame(size=n).select(s=LogNormal().samples(size=size, seed=seed))
    assert result.height == n
    assert result.schema["s"] == pl.Array(pl.Float64, size)


def test_samples_seed_reproducible(frame: Callable[..., pl.DataFrame], seed: int) -> None:
    dframe = frame(size=128)
    s1 = dframe.select(s=LogNormal().samples(size=8, seed=seed))["s"]
    s2 = dframe.select(s=LogNormal().samples(size=8, seed=seed))["s"]
    assert_series_equal(s1, s2)


def test_samples_columns_are_not_all_equal(frame: Callable[..., pl.DataFrame], seed: int) -> None:
    size = 8
    dframe = frame(size=512)
    result = dframe.select(s=LogNormal().samples(size=size, seed=seed))["s"]
    columns = [result.arr.get(i) for i in range(size)]
    distinct = {tuple(c.to_list()) for c in columns}
    assert len(distinct) == size


def test_samples_log_moments_close(frame: Callable[..., pl.DataFrame], seed: int) -> None:
    mu, sigma = 0.5, 0.75
    logs = (
        frame(size=4_000).select(s=LogNormal(mu=mu, sigma=sigma).samples(size=16, seed=seed))["s"].arr.explode().log()
    )
    _mean, _std = cast("float", logs.mean()), cast("float", logs.std())
    assert abs(_mean - mu) < 0.05 * sigma
    assert abs(_std - sigma) < 0.05 * sigma


@pytest.mark.parametrize("bad_size", [0, -1])
def test_samples_rejects_non_positive_size(bad_size: int) -> None:
    with pytest.raises(ValueError, match="size must be a positive integer"):
        LogNormal().samples(size=bad_size, seed=0)


def test_samples_null_param_row_is_null_array(seed: int) -> None:
    size = 4
    dframe = pl.DataFrame(
        {"mu": [0.0, None, 1.0], "sigma": [1.0, 2.0, 3.0]},
        schema={"mu": pl.Float64, "sigma": pl.Float64},
    )
    result = dframe.select(s=LogNormal(mu=pl.col("mu"), sigma=pl.col("sigma")).samples(size=size, seed=seed))["s"]
    assert result.dtype == pl.Array(pl.Float64, size)
    # A null param row yields a null array, not an array of inner-null elements.
    assert_series_equal(result.is_null(), pl.Series("s", [False, True, False]))


def test_samples_non_positive_sigma_raises(seed: int) -> None:
    dframe = pl.DataFrame({"mu": [0.0, 1.0], "sigma": [1.0, -2.0]})  # row 1: sigma = -2.0
    with pytest.raises(pl.exceptions.ComputeError, match="sigma must be finite and strictly positive"):
        dframe.select(s=LogNormal(mu=pl.col("mu"), sigma=pl.col("sigma")).samples(size=4, seed=seed))
