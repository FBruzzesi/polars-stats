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
    df = _frame()
    s1 = df.with_columns(b=Bernoulli(p=p).sample(seed=_SEED))["b"]
    s2 = df.with_columns(b=Bernoulli(p=p).sample(seed=_SEED))["b"]
    assert_series_equal(s1, s2)


@pytest.mark.parametrize("p", [0.3, 0.5, 0.8, pl.col("p1"), pl.col("p2")])
def test_sample_different_seeds_differ(p: float | pl.Expr) -> None:
    df = _frame()
    s1 = df.with_columns(b=Bernoulli(p=p).sample(seed=123))["b"]
    s2 = df.with_columns(b=Bernoulli(p=p).sample(seed=_SEED))["b"]
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
    df = pl.DataFrame({"p": p_values})
    result = df.with_columns(b=Bernoulli(p=pl.col("p")).sample(seed=_SEED))
    assert result["b"].to_list() == p_values


@pytest.mark.parametrize("bad_p", [-0.1, 1.5, float("nan")])
def test_sample_with_column_p_out_of_range_raises(bad_p: float) -> None:
    # Plugin-side errors are always wrapped in ComputeError by the polars FFI
    # layer (the underlying PolarsError::InvalidOperation variant is lost at
    # the plugin boundary). The message is preserved.
    df = pl.DataFrame({"p": [0.5, bad_p, 0.7]})
    with pytest.raises(pl.exceptions.ComputeError, match="p must be in"):
        df.with_columns(b=Bernoulli(p=pl.col("p")).sample(seed=_SEED))


def test_sample_with_null_p_row_is_null() -> None:
    df = pl.DataFrame({"p": [0.5, None, 0.5]}, schema={"p": pl.Float64})
    result = df.with_columns(Bernoulli(p=pl.col("p")).sample(seed=_SEED))
    assert result["p"].item(1) is None


def test_sample_multi_columns() -> None:
    df = _frame()
    bernoulli_expr = Bernoulli(p=pl.col("p1", "p2"))
    result = df.with_columns(bernoulli_expr.sample(seed=_SEED).name.suffix("_b"))
    assert result.schema == pl.Schema(df.schema | {"p1_b": pl.UInt8(), "p2_b": pl.UInt8()})
