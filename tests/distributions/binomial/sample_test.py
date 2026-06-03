from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_series_equal, assert_series_not_equal

from polars_stats import Binomial
from tests.distributions.binomial.conftest import N_TRIALS

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize("size", [0, 100, 1_000])
def test_sample_basic_properties(size: int, frame: Callable[..., pl.DataFrame], seed: int) -> None:
    result = frame(size=size).lazy().with_columns(b=Binomial(N_TRIALS, 0.5).sample(seed=seed))
    assert result.collect_schema()["b"] == pl.UInt64
    collected = result.collect()
    assert collected.height == size
    assert collected["b"].is_between(0, N_TRIALS).all()


@pytest.mark.parametrize("p", [0.0, 0.3, 0.5, 0.8, 1.0, pl.col("p1")])
def test_sample_seed_reproducible(p: float | pl.Expr, frame: Callable[..., pl.DataFrame], seed: int) -> None:
    dframe = frame()
    s1 = dframe.with_columns(b=Binomial(pl.col("n1"), p).sample(seed=seed))["b"]
    s2 = dframe.with_columns(b=Binomial(pl.col("n1"), p).sample(seed=seed))["b"]
    assert_series_equal(s1, s2)


def test_sample_different_seeds_differ(frame: Callable[..., pl.DataFrame], seed: int) -> None:
    dframe = frame()
    s1 = dframe.with_columns(b=Binomial(N_TRIALS, 0.5).sample(seed=123))["b"]
    s2 = dframe.with_columns(b=Binomial(N_TRIALS, 0.5).sample(seed=seed))["b"]
    assert_series_not_equal(s1, s2)


@pytest.mark.parametrize(("p", "expected"), [(0.0, 0), (1.0, N_TRIALS)])
def test_sample_extreme_p_is_deterministic(p: float, expected: int, frame: Callable[..., pl.DataFrame]) -> None:
    # p=0 → 0 successes; p=1 → all n successes, regardless of the draw.
    result = frame().with_columns(b=Binomial(N_TRIALS, p).sample(seed=7))
    assert result["b"].eq(expected).all()


@pytest.mark.parametrize("p", [0.2, 0.5, 0.8])
def test_sample_mean_close_to_np_for_large_n(p: float, frame: Callable[..., pl.DataFrame]) -> None:
    n_rows, tolerance = 100_000, 0.05
    result = frame(n_rows).with_columns(b=Binomial(N_TRIALS, p).sample(seed=123))
    observed = result["b"].mean()
    assert isinstance(observed, float)
    assert abs(observed - N_TRIALS * p) < tolerance


@pytest.mark.parametrize("p", [-0.1, 1.5, float("nan")])
def test_sample_invalid_p_raises(p: float, frame: Callable[..., pl.DataFrame], seed: int) -> None:
    dframe = frame(size=8)
    with pytest.raises(pl.exceptions.ComputeError, match="p must be in"):
        dframe.with_columns(b=Binomial(N_TRIALS, p).sample(seed=seed))


def test_sample_negative_n_raises(seed: int) -> None:
    dframe = pl.DataFrame({"n": [5, -1, 3]}, schema={"n": pl.Int64})
    with pytest.raises(pl.exceptions.ComputeError, match="n must be a non-negative integer"):
        dframe.with_columns(b=Binomial(pl.col("n"), 0.5).sample(seed=seed))


def test_sample_with_null_param_row_is_null(seed: int) -> None:
    dframe = pl.DataFrame({"p": [0.5, None, 0.5]}, schema={"p": pl.Float64})
    result = dframe.with_columns(b=Binomial(N_TRIALS, pl.col("p")).sample(seed=seed))
    assert_series_equal(result["b"].is_null(), pl.Series("b", [False, True, False]))


def test_sample_with_str_params_matches_col(frame: Callable[..., pl.DataFrame], seed: int) -> None:
    dframe = frame()
    s_str = dframe.with_columns(b=Binomial("n1", "p1").sample(seed=seed))["b"]
    s_col = dframe.with_columns(b=Binomial(pl.col("n1"), pl.col("p1")).sample(seed=seed))["b"]
    assert_series_equal(s_str, s_col)


def test_sample_in_group_by_draws_per_group(seed: int) -> None:
    local_rng = np.random.default_rng(seed=seed)
    n_per_group, n_groups = 200, 40
    dframe = pl.DataFrame(
        {
            "group": np.repeat(np.arange(n_groups), n_per_group),
            "probas": local_rng.uniform(0.01, 0.99, size=n_groups * n_per_group),
        },
    )
    agg = (
        dframe.group_by("group", maintain_order=True)
        .agg(
            sum_const=Binomial(N_TRIALS, 0.5).sample(seed=seed).sum(),
            sum_varying=Binomial(N_TRIALS, pl.col("probas")).sample(seed=seed).sum(),
            size=pl.len(),
        )
        .sort("group")
    )
    assert agg.get_column("size").eq(n_per_group).all()
    # Constant p + constant seed + identical per-group indices => identical sums across groups.
    assert len(set(agg.get_column("sum_const").to_list())) == 1
    # Per-row p varies across groups, so per-group sums must vary too.
    assert len(set(agg.get_column("sum_varying").to_list())) > 1


def test_sample_unseeded_produces_variability(frame: Callable[..., pl.DataFrame]) -> None:
    dframe = frame(size=512)
    s1 = dframe.with_columns(b=Binomial(N_TRIALS, 0.5).sample(seed=None))["b"]
    s2 = dframe.with_columns(b=Binomial(N_TRIALS, 0.5).sample(seed=None))["b"]
    assert_series_not_equal(s1, s2)
