from __future__ import annotations

from typing import TYPE_CHECKING, cast

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import Binomial
from tests._polars_compat import arr_explode
from tests.distributions.binomial.conftest import N_TRIALS

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize("size", [1, 4, 16])
def test_samples_shape_and_dtype(size: int, frame: Callable[..., pl.DataFrame], seed: int) -> None:
    n_rows = 32
    result = frame(size=n_rows).select(s=Binomial(N_TRIALS, 0.5).samples(size=size, seed=seed))
    assert result.height == n_rows
    assert result.schema["s"] == pl.Array(pl.UInt64, size)


def test_samples_seed_reproducible(frame: Callable[..., pl.DataFrame], seed: int) -> None:
    dframe = frame(size=128)
    s1 = dframe.select(s=Binomial(N_TRIALS, 0.4).samples(size=8, seed=seed))["s"]
    s2 = dframe.select(s=Binomial(N_TRIALS, 0.4).samples(size=8, seed=seed))["s"]
    assert_series_equal(s1, s2)


def test_samples_columns_are_not_all_equal(frame: Callable[..., pl.DataFrame], seed: int) -> None:
    size = 8
    dframe = frame(size=512)
    result = dframe.select(s=Binomial(N_TRIALS, 0.5).samples(size=size, seed=seed))["s"]
    columns = [result.arr.get(i) for i in range(size)]
    distinct = {tuple(c.to_list()) for c in columns}
    assert len(distinct) == size


def test_samples_mean_close_to_np_for_large_total(frame: Callable[..., pl.DataFrame], seed: int) -> None:
    n_rows, size, p = 4_000, 16, 0.3
    tolerance = 0.05
    result = arr_explode(frame(n_rows).select(s=Binomial(N_TRIALS, p).samples(size=size, seed=seed))["s"])
    mean = cast("float", result.mean())
    assert abs(mean - N_TRIALS * p) < tolerance


@pytest.mark.parametrize("bad_size", [0, -1])
def test_samples_rejects_non_positive_size(bad_size: int) -> None:
    with pytest.raises(ValueError, match="size must be a positive integer"):
        Binomial(N_TRIALS, 0.5).samples(size=bad_size, seed=0)


def test_samples_null_param_row_is_null_array(seed: int) -> None:
    size = 4
    dframe = pl.DataFrame({"p": [0.5, None, 0.5]}, schema={"p": pl.Float64})
    result = dframe.select(s=Binomial(N_TRIALS, pl.col("p")).samples(size=size, seed=seed))["s"]
    assert result.dtype == pl.Array(pl.UInt64, size)
    assert_series_equal(result.is_null(), pl.Series("s", [False, True, False]))
