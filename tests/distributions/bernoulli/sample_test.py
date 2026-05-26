from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_series_equal, assert_series_not_equal

from polars_stats import Bernoulli

if TYPE_CHECKING:
    from collections.abc import Callable


_EXTREME_P = [0.0, 1.0, 0.0, 1.0, 0.0]


@pytest.mark.parametrize("size", [0, 100, 1_000])
def test_sample_basic_properties(
    size: int,
    frame: Callable[..., pl.DataFrame],
    seed: int,
) -> None:
    result = frame(size=size).lazy().with_columns(bernoulli=Bernoulli(p=0.5).sample(seed=seed))
    assert result.collect_schema()["bernoulli"] == pl.Boolean
    assert set(result.collect()["bernoulli"].to_list()) <= {0, 1}
    assert result.collect().height == size


@pytest.mark.parametrize("p", [0.0, 0.3, 0.5, 0.8, 1.0, pl.col("p1"), pl.col("p2")])
def test_sample_seed_reproducible(
    p: float | pl.Expr,
    frame: Callable[..., pl.DataFrame],
    seed: int,
) -> None:
    dframe = frame()
    s1 = dframe.with_columns(b=Bernoulli(p=p).sample(seed=seed))["b"]
    s2 = dframe.with_columns(b=Bernoulli(p=p).sample(seed=seed))["b"]
    assert_series_equal(s1, s2)


@pytest.mark.parametrize("p", [0.3, 0.5, 0.8, pl.col("p1"), pl.col("p2")])
def test_sample_different_seeds_differ(
    p: float | pl.Expr,
    frame: Callable[..., pl.DataFrame],
    seed: int,
) -> None:
    dframe = frame()
    s1 = dframe.with_columns(b=Bernoulli(p=p).sample(seed=123))["b"]
    s2 = dframe.with_columns(b=Bernoulli(p=p).sample(seed=seed))["b"]
    assert_series_not_equal(s1, s2)


@pytest.mark.parametrize("p", [0.0, 1.0])
def test_sample_extreme_p(p: float, frame: Callable[..., pl.DataFrame]) -> None:
    result = frame().with_columns(b=Bernoulli(p=p).sample(seed=7))
    assert result["b"].eq(bool(p)).all()


@pytest.mark.parametrize("p", [0.0, 0.3, 0.5, 0.8, 1.0])
def test_sample_mean_close_to_p_for_large_n(p: float, frame: Callable[..., pl.DataFrame]) -> None:
    n = 100_000
    tolerance = 0.01
    result = frame(n).with_columns(b=Bernoulli(p=p).sample(seed=123))
    observed = result["b"].mean()
    assert isinstance(observed, float)
    assert abs(observed - p) < tolerance


@pytest.mark.parametrize("p", [-0.1, 1.5, float("nan")])
def test_sample_invalid_float_p_raises(p: float, frame: Callable[..., pl.DataFrame], seed: int) -> None:
    # Float-`p` validation is delegated to the plugin so the error surface
    # matches the column-`p` path: a ComputeError at evaluation time, not a
    # ValueError at construction time.
    dframe = frame(size=8)
    with pytest.raises(pl.exceptions.ComputeError, match="p must be in"):
        dframe.with_columns(b=Bernoulli(p=p).sample(seed=seed))


@pytest.mark.parametrize(
    ("dframe", "p_arg"),
    [
        pytest.param(pl.DataFrame({"p": _EXTREME_P}), pl.col("p"), id="expr"),
        pytest.param(pl.DataFrame({"p": _EXTREME_P}), "p", id="str"),
        pytest.param(
            # Series path: frame intentionally has no "p" column so we exercise
            # the Series-as-literal path (not an accidental column lookup).
            pl.DataFrame({"x": list(range(len(_EXTREME_P)))}),
            pl.Series("p", _EXTREME_P, dtype=pl.Float64),
            id="series",
        ),
    ],
)
def test_sample_with_into_expr_column_p_per_row(
    dframe: pl.DataFrame,
    p_arg: object,
    seed: int,
) -> None:
    result = dframe.with_columns(b=Bernoulli(p=p_arg).sample(seed=seed))  # type: ignore[arg-type]
    assert result["b"].to_list() == [bool(v) for v in _EXTREME_P]


def test_sample_with_str_p_matches_col(frame: Callable[..., pl.DataFrame], seed: int) -> None:
    # Passing the column name as `str` must behave identically to `pl.col(name)`.
    dframe = frame()
    s_str = dframe.with_columns(b=Bernoulli(p="p1").sample(seed=seed))["b"]
    s_col = dframe.with_columns(b=Bernoulli(p=pl.col("p1")).sample(seed=seed))["b"]
    assert_series_equal(s_str, s_col)


def test_sample_with_str_p_missing_column_raises(seed: int) -> None:
    # Column resolution happens at expression evaluation, not in __init__.
    dframe = pl.DataFrame({"p": [0.5, 0.5]})
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        dframe.with_columns(b=Bernoulli(p="does_not_exist").sample(seed=seed))


def test_sample_with_series_p_seed_reproducible(rng: np.random.Generator, seed: int) -> None:
    n = 1_000
    p_series = pl.Series("p", rng.uniform(0.01, 0.99, size=n))
    dframe = pl.DataFrame({"x": list(range(n))})
    s1 = dframe.with_columns(b=Bernoulli(p=p_series).sample(seed=seed))["b"]
    s2 = dframe.with_columns(b=Bernoulli(p=p_series).sample(seed=seed))["b"]
    assert_series_equal(s1, s2)


@pytest.mark.parametrize("bad_p", [-0.1, 1.5, float("nan")])
def test_sample_with_column_p_out_of_range_raises(bad_p: float, seed: int) -> None:
    # Plugin-side errors are always wrapped in ComputeError by the polars FFI
    # layer (the underlying PolarsError::InvalidOperation variant is lost at
    # the plugin boundary). The message is preserved.
    dframe = pl.DataFrame({"p": [0.5, bad_p, 0.7]})
    with pytest.raises(pl.exceptions.ComputeError, match="p must be in"):
        dframe.with_columns(b=Bernoulli(p=pl.col("p")).sample(seed=seed))


def test_sample_with_null_p_row_is_null(seed: int) -> None:
    dframe = pl.DataFrame({"p": [0.5, None, 0.5]}, schema={"p": pl.Float64})
    result = dframe.with_columns(Bernoulli(p=pl.col("p")).sample(seed=seed))
    assert result["p"].item(1) is None


def test_sample_multi_columns(frame: Callable[..., pl.DataFrame], seed: int) -> None:
    dframe = frame()
    bernoulli_expr = Bernoulli(p=pl.col("p1", "p2"))
    result = dframe.with_columns(bernoulli_expr.sample(seed=seed).name.suffix("_b"))
    assert result.schema == pl.Schema(dframe.schema | {"p1_b": pl.Boolean(), "p2_b": pl.Boolean()})


@pytest.mark.parametrize(
    "verb",
    [
        pytest.param(lambda df, expr: df.select(b=expr), id="select"),
        pytest.param(lambda df, expr: df.with_columns(b=expr), id="with_columns"),
    ],
)
def test_sample_returns_full_length_boolean(
    verb: Callable[[pl.DataFrame, pl.Expr], pl.DataFrame],
    frame: Callable[..., pl.DataFrame],
    seed: int,
) -> None:
    # Both verbs must return a length-n Boolean column with both 0 and 1
    # present (i.e. no constant-broadcast collapse).
    n = 256
    dframe = frame(size=n)
    result = verb(dframe, Bernoulli(p=0.5).sample(seed=seed))
    assert result.height == n

    series = result.get_column("b")
    assert series.dtype == pl.Boolean
    assert set(series.unique().to_list()) == {0, 1}


def test_sample_in_group_by_draws_per_group(seed: int) -> None:
    # Under genuine elementwise semantics, draws depend only on (seed, row index, p).
    # With a constant p and a constant seed, every group sees the same per-row indices
    # 0..n-1 and so produces an identical bit pattern — variability across groups now
    # comes from p varying per row. We test:
    #   * `sum_p05` (constant p): all groups produce the same sum (deterministic);
    #   * `sum_probas` (per-row p): groups differ;
    #   * per-group size matches `n_per_group`.
    local_rng = np.random.default_rng(seed=seed)
    n_per_group = 200
    n_groups = 50
    dframe = pl.DataFrame(
        {
            "group": np.repeat(np.arange(n_groups), n_per_group),
            "probas": local_rng.uniform(0.01, 0.99, size=n_groups * n_per_group),
        },
    )

    agg = (
        dframe.group_by("group", maintain_order=True)
        .agg(
            sum_p05=Bernoulli(p=0.5).sample(seed=seed).sum(),
            sum_probas=Bernoulli(p=pl.col("probas")).sample(seed=seed).sum(),
            size=pl.len(),
        )
        .sort("group")
    )
    assert agg.get_column("size").eq(n_per_group).all()

    sums_p05 = agg.get_column("sum_p05")
    # Constant p + constant seed + same per-group indices => identical sums.
    assert len(set(sums_p05.to_list())) == 1
    assert sums_p05.is_between(0, n_per_group).all()

    sums_probas = agg.get_column("sum_probas")
    # Per-row p varies across groups, so per-group sums must vary too.
    assert len(set(sums_probas.to_list())) > 1
    assert sums_probas.is_between(0, n_per_group).all()


def test_sample_over_partitions_draws_per_partition(seed: int) -> None:
    # `over` evaluates the expression per window and aligns back to row order.
    # Requirements:
    #   1. full-length output;
    #   2. seed reproducibility across two identical calls;
    #   3. partitions are NOT identical sequences. If polars reseeds the RNG per partition,
    #      every group would receive the same draw vector. With p=0.5 and 200 draws per
    #      group the per-group sums would then collide on a single value; check they vary.
    n_per_group = 200
    n_groups = 20
    dframe = pl.DataFrame({"group": np.repeat(np.arange(n_groups), n_per_group)})
    s1 = dframe.with_columns(b=Bernoulli(p=0.5).sample(seed=seed).over("group"))["b"]
    s2 = dframe.with_columns(b=Bernoulli(p=0.5).sample(seed=seed).over("group"))["b"]
    assert s1.len() == n_groups * n_per_group
    assert set(s1.unique().to_list()) == {0, 1}
    assert_series_equal(s1, s2)

    per_group_sums = (
        dframe.with_columns(b=Bernoulli(p=0.5).sample(seed=seed).over("group"))
        .group_by("group")
        .agg(pl.col("b").sum())
        .get_column("b")
        .to_list()
    )
    assert len(set(per_group_sums)) > 1


def test_sample_unseeded_produces_variability(frame: Callable[..., pl.DataFrame]) -> None:
    # Without a seed, two successive draws on the same frame should not collide. They are not
    # required to be statistically independent, but `assert_series_not_equal` is sufficient
    # to catch the pathological case where `seed=None` accidentally became deterministic.
    dframe = frame(size=512)
    s1 = dframe.with_columns(b=Bernoulli(p=0.5).sample(seed=None))["b"]
    s2 = dframe.with_columns(b=Bernoulli(p=0.5).sample(seed=None))["b"]
    assert_series_not_equal(s1, s2)
