from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.testing import assert_series_equal, assert_series_not_equal

from polars_stats import DiscreteUniform

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize("size", [0, 100, 1_000])
def test_sample_basic_properties(
    size: int,
    frame: Callable[..., pl.DataFrame],
    seed: int,
) -> None:
    result = frame(size=size).lazy().with_columns(du=DiscreteUniform(min=1, max=6).sample(seed=seed))
    assert result.collect_schema()["du"] == pl.Int64
    assert result.collect().height == size


@pytest.mark.parametrize("bounds", [(1, 6), (-5, 9), (pl.col("lo"), pl.col("hi"))])
def test_sample_seed_reproducible(
    bounds: tuple[int | pl.Expr, int | pl.Expr],
    frame: Callable[..., pl.DataFrame],
    seed: int,
) -> None:
    dframe = frame()
    s1 = dframe.with_columns(k=DiscreteUniform(min=bounds[0], max=bounds[1]).sample(seed=seed))["k"]
    s2 = dframe.with_columns(k=DiscreteUniform(min=bounds[0], max=bounds[1]).sample(seed=seed))["k"]
    assert_series_equal(s1, s2)


def test_sample_different_seeds_differ(frame: Callable[..., pl.DataFrame], seed: int) -> None:
    dframe = frame(size=128)
    s1 = dframe.with_columns(k=DiscreteUniform(min=1, max=6).sample(seed=123))["k"]
    s2 = dframe.with_columns(k=DiscreteUniform(min=1, max=6).sample(seed=seed))["k"]
    assert_series_not_equal(s1, s2)


def test_sample_support_is_the_inclusive_range(frame: Callable[..., pl.DataFrame]) -> None:
    lo, hi = 3, 8
    result = frame().with_columns(k=DiscreteUniform(min=lo, max=hi).sample(seed=7))
    assert ((result["k"] >= lo) & (result["k"] <= hi)).all()
    # A wide enough sweep should see both inclusive bounds themselves.
    wide = frame(10_000).with_columns(k=DiscreteUniform(min=lo, max=hi).sample(seed=7))
    assert set(wide["k"].unique()) == set(range(lo, hi + 1))


def test_sample_point_mass_is_always_the_single_point(frame: Callable[..., pl.DataFrame]) -> None:
    point = 4
    result = frame().with_columns(k=DiscreteUniform(min=point, max=point).sample(seed=7))
    assert (result["k"] == point).all()


def test_sample_scalar_fast_path_matches_per_row() -> None:
    # The constant-parameter fast path and the per-row plugin must agree bit for bit for the same
    # parameterisation: one routes through kwargs, the other through columns.
    n = 512
    df = pl.DataFrame({"lo": [2] * n, "hi": [11] * n})
    scalar = df.select(k=DiscreteUniform(min=2, max=11).sample(seed=9))["k"]
    per_row = df.select(k=DiscreteUniform(min=pl.col("lo"), max=pl.col("hi")).sample(seed=9))["k"]
    assert_series_equal(scalar, per_row)


def test_sample_mean_close_to_midpoint_for_large_n(frame: Callable[..., pl.DataFrame]) -> None:
    n = 100_000
    result = frame(n).with_columns(k=DiscreteUniform(min=1, max=6).sample(seed=123))
    observed = result["k"].mean()
    assert isinstance(observed, float)
    assert abs(observed - 3.5) < 0.01 * 5


def test_sample_null_bound_row_is_null(seed: int) -> None:
    dframe = pl.DataFrame({"lo": [1, None, 1], "hi": [6, 6, 6]}, schema={"lo": pl.Int64, "hi": pl.Int64})
    result = dframe.select(k=DiscreteUniform(min=pl.col("lo"), max=pl.col("hi")).sample(seed=seed))["k"]
    assert result[0] is not None
    assert result[1] is None
