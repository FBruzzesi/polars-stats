from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_series_equal, assert_series_not_equal

from polars_stats import Bernoulli

_SEED = 42
_DEFAULT_SIZE = 1000
RNG = np.random.default_rng(seed=_SEED)


def _frame(size: int = _DEFAULT_SIZE) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": list(range(size)),
            "p1": RNG.uniform(0.01, 0.99, size=size),
            "p2": RNG.uniform(0.01, 0.99, size=size),
        },
    )


@pytest.mark.parametrize("size", [0, 100, 1_000])
def test_sample_basice_properties(size: int) -> None:
    result = _frame(size=size).lazy().with_columns(bernoulli=Bernoulli(p=0.5).sample(seed=_SEED))
    assert result.collect_schema()["bernoulli"] == pl.UInt8
    assert set(result.collect()["bernoulli"].to_list()) <= {0, 1}
    assert result.collect().height == size


@pytest.mark.parametrize("p", [0.0, 0.3, 0.5, 0.8, 1.0, pl.col("p1"), pl.col("p2")])
def test_sample_seed_reproducible(p: float | pl.Expr) -> None:
    dframe = _frame()
    s1 = dframe.with_columns(b=Bernoulli(p=p).sample(seed=_SEED))["b"]
    s2 = dframe.with_columns(b=Bernoulli(p=p).sample(seed=_SEED))["b"]
    assert_series_equal(s1, s2)


@pytest.mark.parametrize("p", [0.3, 0.5, 0.8, pl.col("p1"), pl.col("p2")])
def test_sample_different_seeds_differ(p: float | pl.Expr) -> None:
    dframe = _frame()
    s1 = dframe.with_columns(b=Bernoulli(p=p).sample(seed=123))["b"]
    s2 = dframe.with_columns(b=Bernoulli(p=p).sample(seed=_SEED))["b"]
    assert_series_not_equal(s1, s2)


@pytest.mark.parametrize("p", [0.0, 1.0])
def test_sample_extreme_p(p: float) -> None:
    result = _frame().with_columns(b=Bernoulli(p=p).sample(seed=7))
    assert result["b"].eq(p).all()


@pytest.mark.parametrize("p", [0.0, 0.3, 0.5, 0.8, 1.0])
def test_sample_mean_close_to_p_for_large_n(p: float) -> None:
    n = 100_000
    tolerance = 0.01
    result = _frame(n).with_columns(b=Bernoulli(p=p).sample(seed=123))
    observed = result["b"].mean()
    assert isinstance(observed, float)
    assert abs(observed - p) < tolerance


@pytest.mark.parametrize("p", [-0.1, 1.5])
def test_construct_invalid_p(p: float) -> None:
    with pytest.raises(ValueError, match="p must be in"):
        Bernoulli(p=p)


def test_sample_with_column_p_per_row() -> None:
    p_values = [0.0, 1.0, 0.0, 1.0, 0.0]
    dframe = pl.DataFrame({"p": p_values})
    result = dframe.with_columns(b=Bernoulli(p=pl.col("p")).sample(seed=_SEED))
    assert result["b"].to_list() == p_values


@pytest.mark.parametrize("bad_p", [-0.1, 1.5, float("nan")])
def test_sample_with_column_p_out_of_range_raises(bad_p: float) -> None:
    # Plugin-side errors are always wrapped in ComputeError by the polars FFI
    # layer (the underlying PolarsError::InvalidOperation variant is lost at
    # the plugin boundary). The message is preserved.
    dframe = pl.DataFrame({"p": [0.5, bad_p, 0.7]})
    with pytest.raises(pl.exceptions.ComputeError, match="p must be in"):
        dframe.with_columns(b=Bernoulli(p=pl.col("p")).sample(seed=_SEED))


def test_sample_with_null_p_row_is_null() -> None:
    dframe = pl.DataFrame({"p": [0.5, None, 0.5]}, schema={"p": pl.Float64})
    result = dframe.with_columns(Bernoulli(p=pl.col("p")).sample(seed=_SEED))
    assert result["p"].item(1) is None


def test_sample_multi_columns() -> None:
    dframe = _frame()
    bernoulli_expr = Bernoulli(p=pl.col("p1", "p2"))
    result = dframe.with_columns(bernoulli_expr.sample(seed=_SEED).name.suffix("_b"))
    assert result.schema == pl.Schema(dframe.schema | {"p1_b": pl.UInt8(), "p2_b": pl.UInt8()})


def test_sample_in_select_returns_full_length() -> None:
    n = 256
    dframe = _frame(size=n)
    result = dframe.select(b=Bernoulli(p=0.5).sample(seed=_SEED))
    assert result.height == n

    series = result.get_column("b")
    assert series.dtype == pl.UInt8
    assert set(series.unique().to_list()) == {0, 1}


def test_sample_in_with_columns_is_not_constant_broadcast() -> None:
    n = 256
    frame = _frame(size=n)
    result = frame.with_columns(b=Bernoulli(p=0.5).sample(seed=_SEED))
    assert result.height == n

    series = result.get_column("b")
    assert series.dtype == pl.UInt8
    assert set(series.unique().to_list()) == {0, 1}


def test_sample_in_group_by_draws_per_group() -> None:
    # Each group must receive its own draw sequence: aggregating Bernoulli(0.5)
    # samples per group should produce variability in the per-group sums and
    # hit both extremes of the range (not a single broadcast value).
    rng = np.random.default_rng(seed=_SEED)
    n_per_group = 200
    n_groups = 50
    dframe = pl.DataFrame(
        {
            "group": np.repeat(np.arange(n_groups), n_per_group),
            "probas": rng.uniform(0.01, 0.99, size=n_groups * n_per_group),
        },
    )

    agg = (
        dframe.group_by("group", maintain_order=True)
        .agg(
            sum_p05=Bernoulli(p=0.5).sample(seed=_SEED).sum(),
            sum_probas=Bernoulli(p=pl.col("probas")).sample(seed=_SEED).sum(),
            size=pl.len(),
        )
        .sort("group")
    )
    assert agg.get_column("size").eq(n_per_group).all()

    sums = agg.get_column("sum_p05")
    # With 200 fair-coin draws per group the sum is ~Binomial(200, 0.5);
    # P(all 50 groups identical) is astronomically small.
    assert len(set(sums.to_list())) > 1
    # And no group should be all-0 or all-200.
    assert sums.is_between(0, n_per_group).all()


def test_sample_over_partitions_draws_per_partition() -> None:
    # `over` evaluates the expression per window and aligns back to row order.
    # We need: full length output, both 0 and 1 present, and seed reproducibility
    # across two identical calls (the elementwise-optimization concern).
    n_per_group = 200
    n_groups = 20
    dframe = pl.DataFrame({"group": np.repeat(np.arange(n_groups), n_per_group)})
    s1 = dframe.with_columns(b=Bernoulli(p=0.5).sample(seed=_SEED).over("group"))["b"]
    s2 = dframe.with_columns(b=Bernoulli(p=0.5).sample(seed=_SEED).over("group"))["b"]
    assert s1.len() == n_groups * n_per_group
    assert set(s1.unique().to_list()) == {0, 1}
    assert_series_equal(s1, s2)
