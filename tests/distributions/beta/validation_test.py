from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from polars_stats import Beta

if TYPE_CHECKING:
    from collections.abc import Callable

# Every public method must report an invalid shape (`a <= 0`, `b <= 0`, or a non-finite parameter)
# as a ComputeError, not silently compute garbage or null the row. Each method builds the
# distribution in Rust via `statrs`, which is what enforces this consistently across the surface,
# and identically for scalar and column inputs.
_METHODS: dict[str, Callable[[Beta], pl.Expr]] = {
    "pdf": lambda d: d.pdf(pl.col("x")),
    "log_pdf": lambda d: d.log_pdf(pl.col("x")),
    "cdf": lambda d: d.cdf(pl.col("x")),
    "log_cdf": lambda d: d.log_cdf(pl.col("x")),
    "sf": lambda d: d.sf(pl.col("x")),
    "log_sf": lambda d: d.log_sf(pl.col("x")),
    "ppf": lambda d: d.ppf(pl.col("q")),
    "isf": lambda d: d.isf(pl.col("q")),
    "mean": lambda d: d.mean(),
    "variance": lambda d: d.variance(),
    "std": lambda d: d.std(),
    "median": lambda d: d.median(),
    "entropy": lambda d: d.entropy(),
    "sample": lambda d: d.sample(seed=0),
    "samples": lambda d: d.samples(size=4, seed=0),
}


@pytest.mark.parametrize("expr_fn", _METHODS.values(), ids=list(_METHODS))
def test_method_raises_on_non_positive_shape_column(expr_fn: Callable[[Beta], pl.Expr]) -> None:
    df = pl.DataFrame({"a": [2.0, -1.0], "b": [3.0, 2.0], "x": [0.5, 0.5], "q": [0.5, 0.5]})
    dist = Beta(a=pl.col("a"), b=pl.col("b"))  # row 1: a = -1.0
    with pytest.raises(pl.exceptions.ComputeError, match="a and b must be finite and strictly positive"):
        df.select(r=expr_fn(dist))


@pytest.mark.parametrize("expr_fn", _METHODS.values(), ids=list(_METHODS))
def test_method_raises_on_non_positive_shape_scalar(expr_fn: Callable[[Beta], pl.Expr]) -> None:
    df = pl.DataFrame({"x": [0.5], "q": [0.5]})
    dist = Beta(a=2.0, b=-1.0)
    with pytest.raises(pl.exceptions.ComputeError, match="a and b must be finite and strictly positive"):
        df.select(r=expr_fn(dist))


@pytest.mark.parametrize(
    "dist",
    [
        Beta(a=float("inf"), b=1.0),
        Beta(a=pl.repeat(float("inf"), n=pl.len(), dtype=pl.Float64()), b=1.0),
    ],
    ids=["scalar", "column"],
)
def test_infinite_shape_raises(dist: Beta) -> None:
    # Unlike the Normal / LogNormal scale, statrs rejects an infinite Beta shape.
    with pytest.raises(pl.exceptions.ComputeError, match="a and b must be finite and strictly positive"):
        pl.DataFrame({"x": [0.5]}).select(r=dist.pdf(pl.col("x")))
