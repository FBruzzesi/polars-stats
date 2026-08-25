from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.testing import assert_series_equal, assert_series_not_equal

from polars_stats import Geometric

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize("size", [0, 100, 1_000])
def test_sample_basic_properties(
    size: int,
    frame: Callable[..., pl.DataFrame],
    seed: int,
) -> None:
    result = frame(size=size).lazy().with_columns(geometric=Geometric(p=0.3).sample(seed=seed))
    assert result.collect_schema()["geometric"] == pl.UInt64
    assert result.collect().height == size


@pytest.mark.parametrize("p", [0.1, 0.3, 0.8, pl.col("p1"), pl.col("p2")])
def test_sample_seed_reproducible(
    p: float | pl.Expr,
    frame: Callable[..., pl.DataFrame],
    seed: int,
) -> None:
    dframe = frame()
    s1 = dframe.with_columns(k=Geometric(p=p).sample(seed=seed))["k"]
    s2 = dframe.with_columns(k=Geometric(p=p).sample(seed=seed))["k"]
    assert_series_equal(s1, s2)


@pytest.mark.parametrize("p", [0.1, 0.3, 0.8, pl.col("p1"), pl.col("p2")])
def test_sample_different_seeds_differ(
    p: float | pl.Expr,
    frame: Callable[..., pl.DataFrame],
    seed: int,
) -> None:
    dframe = frame()
    s1 = dframe.with_columns(k=Geometric(p=p).sample(seed=123))["k"]
    s2 = dframe.with_columns(k=Geometric(p=p).sample(seed=seed))["k"]
    assert_series_not_equal(s1, s2)


@pytest.mark.parametrize("p", [0.05, 0.3, 0.8])
def test_sample_support_is_the_positive_integers(p: float, frame: Callable[..., pl.DataFrame]) -> None:
    result = frame().with_columns(k=Geometric(p=p).sample(seed=7))
    assert (result["k"] >= 1).all()


def test_sample_extreme_p_is_always_one(frame: Callable[..., pl.DataFrame]) -> None:
    result = frame().with_columns(k=Geometric(p=1.0).sample(seed=7))
    assert (result["k"] == 1).all()


@pytest.mark.parametrize("p", [0.05, 0.3, 0.5, 0.8])
def test_sample_mean_close_to_inverse_p_for_large_n(p: float, frame: Callable[..., pl.DataFrame]) -> None:
    n = 100_000
    tolerance = 0.01 * (1 / p)
    result = frame(n).with_columns(k=Geometric(p=p).sample(seed=123))
    observed = result["k"].mean()
    assert isinstance(observed, float)
    assert abs(observed - 1 / p) < tolerance


def test_sample_null_p_row_is_null(seed: int) -> None:
    dframe = pl.DataFrame({"p": [0.5, None, 0.5]}, schema={"p": pl.Float64})
    result = dframe.select(k=Geometric(p=pl.col("p")).sample(seed=seed))["k"]
    assert result[0] is not None
    assert result[1] is None
