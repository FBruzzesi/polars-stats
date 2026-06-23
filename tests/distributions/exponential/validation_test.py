from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from polars_stats import Exponential

if TYPE_CHECKING:
    from collections.abc import Callable

# Every public method must report an invalid rate (`rate <= 0` or `NaN`) as a ComputeError, not
# silently compute garbage. The deterministic methods all derive from the Rust-validated rate; the
# samplers validate in their own plugin.
_METHODS: dict[str, Callable[[Exponential], pl.Expr]] = {
    "pdf": lambda e: e.pdf(pl.col("x")),
    "log_pdf": lambda e: e.log_pdf(pl.col("x")),
    "cdf": lambda e: e.cdf(pl.col("x")),
    "log_cdf": lambda e: e.log_cdf(pl.col("x")),
    "sf": lambda e: e.sf(pl.col("x")),
    "log_sf": lambda e: e.log_sf(pl.col("x")),
    "ppf": lambda e: e.ppf(pl.col("q")),
    "isf": lambda e: e.isf(pl.col("q")),
    "mean": lambda e: e.mean(),
    "variance": lambda e: e.variance(),
    "std": lambda e: e.std(),
    "median": lambda e: e.median(),
    "entropy": lambda e: e.entropy(),
    "sample": lambda e: e.sample(seed=0),
    "samples": lambda e: e.samples(size=4, seed=0),
}


@pytest.mark.parametrize("expr_fn", _METHODS.values(), ids=list(_METHODS))
def test_method_raises_on_non_positive_rate_column(expr_fn: Callable[[Exponential], pl.Expr]) -> None:
    df = pl.DataFrame({"rate": [1.0, -0.5], "x": [0.5, 0.5], "q": [0.5, 0.5]})  # row 1: rate <= 0
    e = Exponential(rate=pl.col("rate"))
    with pytest.raises(pl.exceptions.ComputeError, match="rate must be strictly positive"):
        df.select(r=expr_fn(e))


@pytest.mark.parametrize("expr_fn", _METHODS.values(), ids=list(_METHODS))
def test_method_raises_on_non_positive_rate_scalar(expr_fn: Callable[[Exponential], pl.Expr]) -> None:
    df = pl.DataFrame({"x": [0.5], "q": [0.5]})
    e = Exponential(rate=-1.0)
    with pytest.raises(pl.exceptions.ComputeError, match="rate must be strictly positive"):
        df.select(r=expr_fn(e))


@pytest.mark.parametrize("expr_fn", _METHODS.values(), ids=list(_METHODS))
def test_method_raises_on_nan_rate_scalar(expr_fn: Callable[[Exponential], pl.Expr]) -> None:
    df = pl.DataFrame({"x": [0.5], "q": [0.5]})
    e = Exponential(rate=float("nan"))
    with pytest.raises(pl.exceptions.ComputeError, match="rate must be strictly positive"):
        df.select(r=expr_fn(e))
