from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_series_equal, assert_series_not_equal

from polars_stats import LogNormal

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize("size", [0, 100, 1_000])
def test_sample_basic_properties(size: int, frame: Callable[..., pl.DataFrame], seed: int) -> None:
    result = frame(size=size).lazy().with_columns(z=LogNormal().sample(seed=seed))
    assert result.collect_schema()["z"] == pl.Float64
    assert result.collect().height == size


def test_sample_is_positive(frame: Callable[..., pl.DataFrame], seed: int) -> None:
    # The LogNormal support is (0, inf): every draw is strictly positive.
    s = frame(size=10_000).with_columns(z=LogNormal(mu=0.0, sigma=1.5).sample(seed=seed))["z"]
    assert s.min() > 0.0  # type: ignore[operator]


def test_sample_seed_reproducible(frame: Callable[..., pl.DataFrame], seed: int) -> None:
    dframe = frame()
    s1 = dframe.with_columns(z=LogNormal(mu=1.0, sigma=2.0).sample(seed=seed))["z"]
    s2 = dframe.with_columns(z=LogNormal(mu=1.0, sigma=2.0).sample(seed=seed))["z"]
    assert_series_equal(s1, s2)


def test_sample_different_seeds_differ(frame: Callable[..., pl.DataFrame], seed: int) -> None:
    dframe = frame()
    s1 = dframe.with_columns(z=LogNormal().sample(seed=123))["z"]
    s2 = dframe.with_columns(z=LogNormal().sample(seed=seed))["z"]
    assert_series_not_equal(s1, s2)


@pytest.mark.parametrize(("mu", "sigma"), [(0.0, 1.0), (-1.0, 0.5), (1.0, 0.75)])
def test_sample_log_moments_close_for_large_n(
    mu: float, sigma: float, frame: Callable[..., pl.DataFrame], seed: int
) -> None:
    # ln(X) is Normal(mu, sigma); checking the log-space moments avoids the heavy-tailed convergence
    # of the raw LogNormal mean / std.
    s = frame(size=200_000).with_columns(z=LogNormal(mu=mu, sigma=sigma).sample(seed=seed))["z"]
    logs = s.log()
    observed_mean = logs.mean()
    observed_std = logs.std()
    assert isinstance(observed_mean, float)
    assert isinstance(observed_std, float)
    assert abs(observed_mean - mu) < 0.02 * sigma
    assert abs(observed_std - sigma) < 0.02 * sigma


def test_sample_column_params_log_moments_per_row(seed: int) -> None:
    size = 200_000
    mu, sigma = 0.5, 0.75
    dframe = pl.DataFrame({"mu": [mu] * size, "sigma": [sigma] * size})
    s = dframe.with_columns(z=LogNormal(mu=pl.col("mu"), sigma=pl.col("sigma")).sample(seed=seed))["z"]
    logs = s.log()
    assert abs(logs.mean() - mu) < 0.02 * sigma  # type: ignore[operator]
    assert abs(logs.std() - sigma) < 0.02 * sigma  # type: ignore[operator]


def test_sample_str_params_match_col(frame: Callable[..., pl.DataFrame], seed: int) -> None:
    dframe = frame()
    s_str = dframe.with_columns(z=LogNormal(mu="mu", sigma="sigma").sample(seed=seed))["z"]
    s_col = dframe.with_columns(z=LogNormal(mu=pl.col("mu"), sigma=pl.col("sigma")).sample(seed=seed))["z"]
    assert_series_equal(s_str, s_col)


def test_sample_null_param_row_is_null(seed: int) -> None:
    dframe = pl.DataFrame(
        {"mu": [0.0, None, -1.0], "sigma": [1.0, 2.0, None]},
        schema={"mu": pl.Float64, "sigma": pl.Float64},
    )
    result = dframe.with_columns(z=LogNormal(mu=pl.col("mu"), sigma=pl.col("sigma")).sample(seed=seed))["z"]
    assert_series_equal(result.is_null(), pl.Series("z", [False, True, True]))


def test_sample_non_positive_sigma_row_raises(seed: int) -> None:
    # An invalid scale on a row raises (no early Python validation). Plugin errors surface as `ComputeError`.
    dframe = pl.DataFrame({"mu": [0.0, 1.0, -1.0], "sigma": [1.0, -2.0, 3.0]})  # row 1: sigma = -2.0
    with pytest.raises(pl.exceptions.ComputeError, match="sigma must be finite and strictly positive"):
        dframe.with_columns(z=LogNormal(mu=pl.col("mu"), sigma=pl.col("sigma")).sample(seed=seed))


def test_sample_in_group_by_draws_per_group(seed: int) -> None:
    n_per_group, n_groups = 200, 20
    dframe = pl.DataFrame({"group": np.repeat(np.arange(n_groups), n_per_group)})
    agg = (
        dframe.group_by("group", maintain_order=True)
        .agg(mean_z=LogNormal().sample(seed=seed).mean(), size=pl.len())
        .sort("group")
    )
    assert agg["size"].eq(n_per_group).all()
    # Constant params + constant seed + same per-group indices => identical per-group means.
    assert len(set(agg["mean_z"].to_list())) == 1


def test_sample_over_partitions_full_length(seed: int) -> None:
    n_per_group, n_groups = 200, 10
    dframe = pl.DataFrame({"group": np.repeat(np.arange(n_groups), n_per_group)})
    s1 = dframe.with_columns(z=LogNormal().sample(seed=seed).over("group"))["z"]
    s2 = dframe.with_columns(z=LogNormal().sample(seed=seed).over("group"))["z"]
    assert s1.len() == n_groups * n_per_group
    assert_series_equal(s1, s2)


def test_sample_unseeded_produces_variability(frame: Callable[..., pl.DataFrame]) -> None:
    dframe = frame(size=512)
    s1 = dframe.with_columns(z=LogNormal().sample(seed=None))["z"]
    s2 = dframe.with_columns(z=LogNormal().sample(seed=None))["z"]
    assert_series_not_equal(s1, s2)


def test_sample_rows_are_independent_no_aliasing(seed: int) -> None:
    # Each row derives its stream from `(seed, row_index)`, so even with constant params and a fixed
    # seed every row must draw an independent value.
    size = 50_000
    s = pl.DataFrame({"x": range(size)}).with_columns(z=LogNormal().sample(seed=seed))["z"]
    assert s.n_unique() == size
