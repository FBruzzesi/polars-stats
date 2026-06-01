from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_series_equal, assert_series_not_equal

from polars_stats import Uniform

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize("size", [0, 100, 1_000])
def test_sample_basic_properties(size: int, frame: Callable[..., pl.DataFrame], seed: int) -> None:
    result = frame(size=size).lazy().with_columns(u=Uniform(min=0.0, max=1.0).sample(seed=seed))
    assert result.collect_schema()["u"] == pl.Float64
    assert result.collect().height == size


@pytest.mark.parametrize(("mn", "mx"), [(0.0, 1.0), (-3.0, 2.0), (10.0, 11.0)])
def test_sample_within_bounds(mn: float, mx: float, frame: Callable[..., pl.DataFrame], seed: int) -> None:
    s = frame(size=5_000).with_columns(u=Uniform(min=mn, max=mx).sample(seed=seed))["u"].to_numpy()
    assert s.min() >= mn
    assert s.max() <= mx


def test_sample_seed_reproducible(frame: Callable[..., pl.DataFrame], seed: int) -> None:
    dframe = frame()
    s1 = dframe.with_columns(u=Uniform(min=0.0, max=1.0).sample(seed=seed))["u"]
    s2 = dframe.with_columns(u=Uniform(min=0.0, max=1.0).sample(seed=seed))["u"]
    assert_series_equal(s1, s2)


def test_sample_different_seeds_differ(frame: Callable[..., pl.DataFrame], seed: int) -> None:
    dframe = frame()
    s1 = dframe.with_columns(u=Uniform(min=0.0, max=1.0).sample(seed=123))["u"]
    s2 = dframe.with_columns(u=Uniform(min=0.0, max=1.0).sample(seed=seed))["u"]
    assert_series_not_equal(s1, s2)


@pytest.mark.parametrize(("mn", "mx"), [(0.0, 1.0), (-3.0, 2.0), (10.0, 11.0)])
def test_sample_mean_close_to_midpoint_for_large_n(
    mn: float,
    mx: float,
    frame: Callable[..., pl.DataFrame],
    seed: int,
) -> None:
    s = frame(size=200_000).with_columns(u=Uniform(min=mn, max=mx).sample(seed=seed))["u"]
    observed = s.mean()
    assert isinstance(observed, float)
    assert abs(observed - (mn + mx) / 2) < 0.01 * (mx - mn)


def test_sample_column_bounds(frame: Callable[..., pl.DataFrame], seed: int) -> None:
    dframe = frame(size=2_000)
    s = dframe.with_columns(u=Uniform(min=pl.col("lo"), max=pl.col("hi")).sample(seed=seed))
    assert (s["u"] >= s["lo"]).all()
    assert (s["u"] <= s["hi"]).all()


def test_sample_str_bounds_match_col(frame: Callable[..., pl.DataFrame], seed: int) -> None:
    dframe = frame()
    s_str = dframe.with_columns(u=Uniform(min="lo", max="hi").sample(seed=seed))["u"]
    s_col = dframe.with_columns(u=Uniform(min=pl.col("lo"), max=pl.col("hi")).sample(seed=seed))["u"]
    assert_series_equal(s_str, s_col)


def test_sample_null_bound_row_is_null(seed: int) -> None:
    dframe = pl.DataFrame(
        {"min": [0.0, None, -1.0], "max": [1.0, 2.0, None]},
        schema={"min": pl.Float64, "max": pl.Float64},
    )
    result = dframe.with_columns(u=Uniform(min=pl.col("min"), max=pl.col("max")).sample(seed=seed))["u"]
    assert_series_equal(result.is_null(), pl.Series("u", [False, True, True]))


def test_sample_max_le_min_row_raises(seed: int) -> None:
    # An invalid parameterisation on a row raises (no early Python validation), the same way
    # Bernoulli reports an out-of-range `p`. Plugin errors surface as `ComputeError`.
    dframe = pl.DataFrame({"lo": [0.0, 5.0, -1.0], "hi": [1.0, 2.0, 3.0]})  # row 1: hi (2.0) <= lo (5.0)
    with pytest.raises(pl.exceptions.ComputeError, match="max must be strictly greater than min"):
        dframe.with_columns(u=Uniform(min=pl.col("lo"), max=pl.col("hi")).sample(seed=seed))


def test_sample_in_group_by_draws_per_group(seed: int) -> None:
    n_per_group, n_groups = 200, 20
    local_rng = np.random.default_rng(seed=seed)
    dframe = pl.DataFrame(
        {
            "group": np.repeat(np.arange(n_groups), n_per_group),
            "lo": local_rng.uniform(-5.0, 0.0, size=n_groups * n_per_group),
        },
    ).with_columns(hi=pl.col("lo") + 1.0)

    agg = (
        dframe.group_by("group", maintain_order=True)
        .agg(
            mean_unit=Uniform(min=0.0, max=1.0).sample(seed=seed).mean(),
            size=pl.len(),
        )
        .sort("group")
    )
    assert agg["size"].eq(n_per_group).all()
    # Constant bounds + constant seed + same per-group indices => identical per-group means.
    assert len(set(agg["mean_unit"].to_list())) == 1


def test_sample_over_partitions_full_length(seed: int) -> None:
    n_per_group, n_groups = 200, 10
    dframe = pl.DataFrame({"group": np.repeat(np.arange(n_groups), n_per_group)})
    s1 = dframe.with_columns(u=Uniform(min=0.0, max=1.0).sample(seed=seed).over("group"))["u"]
    s2 = dframe.with_columns(u=Uniform(min=0.0, max=1.0).sample(seed=seed).over("group"))["u"]
    assert s1.len() == n_groups * n_per_group
    assert_series_equal(s1, s2)


def test_sample_unseeded_produces_variability(frame: Callable[..., pl.DataFrame]) -> None:
    dframe = frame(size=512)
    s1 = dframe.with_columns(u=Uniform(min=0.0, max=1.0).sample(seed=None))["u"]
    s2 = dframe.with_columns(u=Uniform(min=0.0, max=1.0).sample(seed=None))["u"]
    assert_series_not_equal(s1, s2)
