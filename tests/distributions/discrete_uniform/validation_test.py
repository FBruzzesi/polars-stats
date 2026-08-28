from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from polars_stats import DiscreteUniform

if TYPE_CHECKING:
    from collections.abc import Callable

# Every public method must report `max < min` (or an overflowing width) as a ComputeError, not
# silently compute. All of them derive from the Rust-validated support count; the samplers validate
# in their own plugin. `value` is the evaluation point, so one dict drives both the column-routed
# sweep and the frameless one below; `ppf` / `isf` ignore it and keep a fixed quantile.
_METHODS: dict[str, Callable[[DiscreteUniform, float | pl.Expr], pl.Expr]] = {
    "pmf": lambda d, v: d.pmf(v),
    "log_pmf": lambda d, v: d.log_pmf(v),
    "cdf": lambda d, v: d.cdf(v),
    "log_cdf": lambda d, v: d.log_cdf(v),
    "sf": lambda d, v: d.sf(v),
    "log_sf": lambda d, v: d.log_sf(v),
    "ppf": lambda d, _v: d.ppf(0.5),
    "isf": lambda d, _v: d.isf(0.5),
    "mean": lambda d, _v: d.mean(),
    "variance": lambda d, _v: d.variance(),
    "std": lambda d, _v: d.std(),
    "median": lambda d, _v: d.median(),
    "entropy": lambda d, _v: d.entropy(),
    "sample": lambda d, _v: d.sample(seed=0),
    "samples": lambda d, _v: d.samples(size=4, seed=0),
    "support_size": lambda d, _v: d.support_size,
}

_INVERTED = "max must be greater than or equal to min"


@pytest.mark.parametrize(("lo", "hi"), [(6, 1), (-2, -7)])
@pytest.mark.parametrize("expr_fn", _METHODS.values(), ids=list(_METHODS))
def test_method_raises_on_inverted_bounds_column(
    lo: int, hi: int, expr_fn: Callable[[DiscreteUniform, float | pl.Expr], pl.Expr]
) -> None:
    df = pl.DataFrame({"lo": [0, lo], "hi": [5, hi], "x": [2.0, 2.0]})
    d = DiscreteUniform(min=pl.col("lo"), max=pl.col("hi"))
    with pytest.raises(pl.exceptions.ComputeError, match=_INVERTED):
        df.select(r=expr_fn(d, pl.col("x")))


@pytest.mark.parametrize("expr_fn", _METHODS.values(), ids=list(_METHODS))
def test_method_raises_on_inverted_bounds_scalar(
    expr_fn: Callable[[DiscreteUniform, float | pl.Expr], pl.Expr],
) -> None:
    df = pl.DataFrame({"x": [2.0]})
    with pytest.raises(pl.exceptions.ComputeError, match=_INVERTED):
        df.select(r=expr_fn(DiscreteUniform(min=6, max=1), pl.col("x")))


# `pl.select` resolves `pl.len()` to 0, so a method whose output height is anchored to the frame
# instead of to its own inputs evaluates zero rows and reports nothing. Constant bounds are validated
# by their own length-1 plugin call, so every method must raise here too.
@pytest.mark.parametrize("expr_fn", _METHODS.values(), ids=list(_METHODS))
def test_method_raises_on_inverted_bounds_without_a_frame(
    expr_fn: Callable[[DiscreteUniform, float | pl.Expr], pl.Expr],
) -> None:
    with pytest.raises(pl.exceptions.ComputeError, match=_INVERTED):
        pl.select(r=expr_fn(DiscreteUniform(min=5, max=0), 2.0))


def test_float_bound_column_is_refused() -> None:
    # The dtype rule is column-level and judged before values: a float bound column is refused even
    # though casting it would be lossless here, because in general it would silently truncate.
    df = pl.DataFrame({"lo": [0.0, 1.0], "hi": [5.0, 6.0]})
    d = DiscreteUniform(min=pl.col("lo"), max=pl.col("hi"))
    with pytest.raises(pl.exceptions.ComputeError, match="bounds must be integer columns"):
        df.select(r=d.mean())


@pytest.mark.parametrize("dtype", [pl.Int8, pl.Int16, pl.Int32, pl.UInt8, pl.UInt32, pl.UInt64])
def test_any_integer_bound_dtype_is_accepted(dtype: pl.DataType) -> None:
    df = pl.DataFrame({"lo": [1, 2], "hi": [6, 7]}, schema={"lo": dtype, "hi": dtype})
    result = df.select(v=DiscreteUniform(min=pl.col("lo"), max=pl.col("hi")).mean())["v"]
    assert result.to_list() == [3.5, 4.5]


def test_unsigned_bound_above_int64_max_raises() -> None:
    # A `UInt64` column holds values `Int64` cannot; the strict cast reports them rather than wrapping.
    df = pl.DataFrame({"lo": [0, 1], "hi": [2**63 + 5, 20]}, schema={"lo": pl.UInt64, "hi": pl.UInt64})
    d = DiscreteUniform(min=pl.col("lo"), max=pl.col("hi"))
    with pytest.raises(pl.exceptions.ComputeError, match="bounds must be integers that fit in i64"):
        df.select(r=d.mean())


@pytest.mark.parametrize("null_bound", ["lo", "hi"])
def test_null_dtype_bound_column_propagates_null(null_bound: str) -> None:
    schema = {"lo": pl.Int64, "hi": pl.Int64} | {null_bound: pl.Null}
    df = pl.DataFrame({"lo": [1, 2], "hi": [6, 7]} | {null_bound: [None, None]}, schema=schema)
    result = df.select(v=DiscreteUniform(min=pl.col("lo"), max=pl.col("hi")).support_size)["v"]
    assert result.to_list() == [None, None]


def test_support_width_overflowing_int64_raises() -> None:
    df = pl.DataFrame({"x": [0.0]})
    d = DiscreteUniform(min=-(2**63), max=2**63 - 1)
    with pytest.raises(pl.exceptions.ComputeError, match="support width"):
        df.select(r=d.mean())


def test_min_equal_max_point_mass_is_valid(unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(pmf=DiscreteUniform(min=3, max=3).pmf(3.0), mean=DiscreteUniform(min=3, max=3).mean())
    assert result["pmf"][0] == pytest.approx(1.0)
    assert result["mean"][0] == pytest.approx(3.0)


@pytest.mark.parametrize(("lo", "hi", "expected"), [(1, 6, 6.0), (-5, 9, 15.0), (3, 3, 1.0), (-20, -10, 11.0)])
def test_support_size_counts_the_inclusive_support(lo: int, hi: int, expected: float, unit_frame: pl.DataFrame) -> None:
    """`max - min + 1`: both bounds inclusive, so a one-point mass counts 1 rather than 0."""
    result = unit_frame.select(v=DiscreteUniform(min=lo, max=hi).support_size).item(0, "v")
    assert result == expected


def test_support_size_propagates_null_in_bounds(bounds_with_null: pl.DataFrame) -> None:
    result = bounds_with_null.select(v=DiscreteUniform(min=pl.col("lo"), max=pl.col("hi")).support_size)["v"]
    assert result.to_list() == [6.0, None, None]
