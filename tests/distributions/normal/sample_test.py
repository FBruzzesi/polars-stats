from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_series_equal, assert_series_not_equal

from polars_stats import Normal

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize("size", [0, 100, 1_000])
def test_sample_basic_properties(size: int, frame: Callable[..., pl.DataFrame], seed: int) -> None:
    result = frame(size=size).lazy().with_columns(z=Normal().sample(seed=seed))
    assert result.collect_schema()["z"] == pl.Float64
    assert result.collect().height == size


def test_sample_seed_reproducible(frame: Callable[..., pl.DataFrame], seed: int) -> None:
    dframe = frame()
    s1 = dframe.with_columns(z=Normal(mu=1.0, sigma=2.0).sample(seed=seed))["z"]
    s2 = dframe.with_columns(z=Normal(mu=1.0, sigma=2.0).sample(seed=seed))["z"]
    assert_series_equal(s1, s2)


def test_sample_different_seeds_differ(frame: Callable[..., pl.DataFrame], seed: int) -> None:
    dframe = frame()
    s1 = dframe.with_columns(z=Normal().sample(seed=123))["z"]
    s2 = dframe.with_columns(z=Normal().sample(seed=seed))["z"]
    assert_series_not_equal(s1, s2)


@pytest.mark.parametrize(("mean", "std"), [(0.0, 1.0), (-3.0, 0.5), (10.0, 5.0)])
def test_sample_moments_close_for_large_n(
    mean: float, std: float, frame: Callable[..., pl.DataFrame], seed: int
) -> None:
    s = frame(size=200_000).with_columns(z=Normal(mu=mean, sigma=std).sample(seed=seed))["z"]
    observed_mean = s.mean()
    observed_std = s.std()
    assert isinstance(observed_mean, float)
    assert isinstance(observed_std, float)
    assert abs(observed_mean - mean) < 0.02 * std
    assert abs(observed_std - std) < 0.02 * std


def test_sample_column_params_moments_per_row(seed: int) -> None:
    # Constant params per row, but a wide frame: the empirical moments must track the parameters.
    size = 200_000
    mean, std = 2.5, 0.75
    dframe = pl.DataFrame({"mu": [mean] * size, "sigma": [std] * size})
    s = dframe.with_columns(z=Normal(mu=pl.col("mu"), sigma=pl.col("sigma")).sample(seed=seed))["z"]
    assert abs(s.mean() - mean) < 0.02 * std  # type: ignore[operator]
    assert abs(s.std() - std) < 0.02 * std  # type: ignore[operator]


def test_sample_str_params_match_col(frame: Callable[..., pl.DataFrame], seed: int) -> None:
    dframe = frame()
    s_str = dframe.with_columns(z=Normal(mu="mu", sigma="sigma").sample(seed=seed))["z"]
    s_col = dframe.with_columns(z=Normal(mu=pl.col("mu"), sigma=pl.col("sigma")).sample(seed=seed))["z"]
    assert_series_equal(s_str, s_col)


def test_sample_null_param_row_is_null(seed: int) -> None:
    dframe = pl.DataFrame(
        {"mu": [0.0, None, -1.0], "sigma": [1.0, 2.0, None]},
        schema={"mu": pl.Float64, "sigma": pl.Float64},
    )
    result = dframe.with_columns(z=Normal(mu=pl.col("mu"), sigma=pl.col("sigma")).sample(seed=seed))["z"]
    assert_series_equal(result.is_null(), pl.Series("z", [False, True, True]))


def test_sample_non_positive_std_row_raises(seed: int) -> None:
    # An invalid scale on a row raises (no early Python validation), the same way Bernoulli reports an
    # out-of-range `p`. Plugin errors surface as `ComputeError`.
    dframe = pl.DataFrame({"mu": [0.0, 1.0, -1.0], "sigma": [1.0, -2.0, 3.0]})  # row 1: sigma = -2.0
    with pytest.raises(pl.exceptions.ComputeError, match="sigma must be finite and strictly positive"):
        dframe.with_columns(z=Normal(mu=pl.col("mu"), sigma=pl.col("sigma")).sample(seed=seed))


def test_sample_in_group_by_draws_per_group(seed: int) -> None:
    n_per_group, n_groups = 200, 20
    dframe = pl.DataFrame({"group": np.repeat(np.arange(n_groups), n_per_group)})
    agg = (
        dframe.group_by("group", maintain_order=True)
        .agg(mean_z=Normal().sample(seed=seed).mean(), size=pl.len())
        .sort("group")
    )
    assert agg["size"].eq(n_per_group).all()
    # Constant params + constant seed + same per-group indices => identical per-group means.
    assert len(set(agg["mean_z"].to_list())) == 1


def test_sample_over_partitions_full_length(seed: int) -> None:
    n_per_group, n_groups = 200, 10
    dframe = pl.DataFrame({"group": np.repeat(np.arange(n_groups), n_per_group)})
    s1 = dframe.with_columns(z=Normal().sample(seed=seed).over("group"))["z"]
    s2 = dframe.with_columns(z=Normal().sample(seed=seed).over("group"))["z"]
    assert s1.len() == n_groups * n_per_group
    assert_series_equal(s1, s2)


def test_sample_unseeded_produces_variability(frame: Callable[..., pl.DataFrame]) -> None:
    dframe = frame(size=512)
    s1 = dframe.with_columns(z=Normal().sample(seed=None))["z"]
    s2 = dframe.with_columns(z=Normal().sample(seed=None))["z"]
    assert_series_not_equal(s1, s2)


def test_sample_rows_are_independent_no_aliasing(seed: int) -> None:
    # Each row derives its stream from `(seed, row_index)`, so even with constant params and a fixed
    # seed every row must draw an independent value. Full uniqueness among Float64 draws is the
    # observable form of the per-row non-aliasing guarantee.
    size = 50_000
    s = pl.DataFrame({"x": range(size)}).with_columns(z=Normal().sample(seed=seed))["z"]
    assert s.n_unique() == size
