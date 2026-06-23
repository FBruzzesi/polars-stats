from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_series_equal, assert_series_not_equal

from polars_stats import Exponential

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize("size", [0, 100, 1_000])
def test_sample_basic_properties(size: int, frame: Callable[..., pl.DataFrame], seed: int) -> None:
    result = frame(size=size).lazy().with_columns(e=Exponential(rate=1.0).sample(seed=seed))
    assert result.collect_schema()["e"] == pl.Float64
    assert result.collect().height == size


@pytest.mark.parametrize("rate_value", [0.5, 1.0, 5.0])
def test_sample_non_negative(rate_value: float, frame: Callable[..., pl.DataFrame], seed: int) -> None:
    s = frame(size=5_000).with_columns(e=Exponential(rate=rate_value).sample(seed=seed))["e"].to_numpy()
    assert s.min() >= 0.0


def test_sample_seed_reproducible(frame: Callable[..., pl.DataFrame], seed: int) -> None:
    dframe = frame()
    s1 = dframe.with_columns(e=Exponential(rate=1.0).sample(seed=seed))["e"]
    s2 = dframe.with_columns(e=Exponential(rate=1.0).sample(seed=seed))["e"]
    assert_series_equal(s1, s2)


def test_sample_different_seeds_differ(frame: Callable[..., pl.DataFrame], seed: int) -> None:
    dframe = frame()
    s1 = dframe.with_columns(e=Exponential(rate=1.0).sample(seed=123))["e"]
    s2 = dframe.with_columns(e=Exponential(rate=1.0).sample(seed=seed))["e"]
    assert_series_not_equal(s1, s2)


@pytest.mark.parametrize("rate_value", [0.5, 1.0, 5.0])
def test_sample_mean_close_to_inverse_rate_for_large_n(
    rate_value: float,
    frame: Callable[..., pl.DataFrame],
    seed: int,
) -> None:
    s = frame(size=200_000).with_columns(e=Exponential(rate=rate_value).sample(seed=seed))["e"]
    observed = s.mean()
    assert isinstance(observed, float)
    assert abs(observed - 1.0 / rate_value) < 0.03 * (1.0 / rate_value)


def test_sample_column_rates(frame: Callable[..., pl.DataFrame], seed: int) -> None:
    dframe = frame(size=2_000)
    s = dframe.with_columns(e=Exponential(rate=pl.col("rate")).sample(seed=seed))
    assert (s["e"] >= 0.0).all()


def test_sample_str_rate_match_col(frame: Callable[..., pl.DataFrame], seed: int) -> None:
    dframe = frame()
    s_str = dframe.with_columns(e=Exponential(rate="rate").sample(seed=seed))["e"]
    s_col = dframe.with_columns(e=Exponential(rate=pl.col("rate")).sample(seed=seed))["e"]
    assert_series_equal(s_str, s_col)


def test_sample_null_rate_row_is_null(seed: int) -> None:
    dframe = pl.DataFrame({"rate": [1.0, None, 2.0]}, schema={"rate": pl.Float64})
    result = dframe.with_columns(e=Exponential(rate=pl.col("rate")).sample(seed=seed))["e"]
    assert_series_equal(result.is_null(), pl.Series("e", [False, True, False]))


def test_sample_non_positive_rate_row_raises(seed: int) -> None:
    # An invalid rate on a row raises (no early Python validation), like Uniform / Bernoulli.
    dframe = pl.DataFrame({"rate": [1.0, -0.5, 2.0]})  # row 1: rate <= 0
    with pytest.raises(pl.exceptions.ComputeError, match="rate must be strictly positive"):
        dframe.with_columns(e=Exponential(rate=pl.col("rate")).sample(seed=seed))


def test_sample_in_group_by_draws_per_group(seed: int) -> None:
    n_per_group, n_groups = 200, 20
    dframe = pl.DataFrame({"group": np.repeat(np.arange(n_groups), n_per_group)})

    agg = (
        dframe.group_by("group", maintain_order=True)
        .agg(
            mean_unit=Exponential(rate=1.0).sample(seed=seed).mean(),
            size=pl.len(),
        )
        .sort("group")
    )
    assert agg["size"].eq(n_per_group).all()
    # Constant rate + constant seed + same per-group indices => identical per-group means.
    assert len(set(agg["mean_unit"].to_list())) == 1


def test_sample_over_partitions_full_length(seed: int) -> None:
    n_per_group, n_groups = 200, 10
    dframe = pl.DataFrame({"group": np.repeat(np.arange(n_groups), n_per_group)})
    s1 = dframe.with_columns(e=Exponential(rate=1.0).sample(seed=seed).over("group"))["e"]
    s2 = dframe.with_columns(e=Exponential(rate=1.0).sample(seed=seed).over("group"))["e"]
    assert s1.len() == n_groups * n_per_group
    assert_series_equal(s1, s2)


def test_sample_unseeded_produces_variability(frame: Callable[..., pl.DataFrame]) -> None:
    dframe = frame(size=512)
    s1 = dframe.with_columns(e=Exponential(rate=1.0).sample(seed=None))["e"]
    s2 = dframe.with_columns(e=Exponential(rate=1.0).sample(seed=None))["e"]
    assert_series_not_equal(s1, s2)


def test_sample_rows_are_independent_no_aliasing(seed: int) -> None:
    # Each row derives its stream from `(seed, row_index)`, so even with a constant rate and a fixed
    # seed every row draws an independent value; full uniqueness is the observable form of that.
    size = 50_000
    s = pl.DataFrame({"x": range(size)}).with_columns(e=Exponential(rate=1.0).sample(seed=seed))["e"]
    assert s.n_unique() == size
