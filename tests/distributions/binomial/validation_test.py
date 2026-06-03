from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from polars_stats import Binomial

if TYPE_CHECKING:
    from collections.abc import Callable

# Every public method must report an invalid parameter as a ComputeError, not silently null the row.
# All deterministic methods build the statrs distribution in their plugin; the sampler validates in
# its own plugin.
_METHODS: dict[str, Callable[[Binomial], pl.Expr]] = {
    "pmf": lambda b: b.pmf(pl.col("x")),
    "log_pmf": lambda b: b.log_pmf(pl.col("x")),
    "cdf": lambda b: b.cdf(pl.col("x")),
    "log_cdf": lambda b: b.log_cdf(pl.col("x")),
    "sf": lambda b: b.sf(pl.col("x")),
    "log_sf": lambda b: b.log_sf(pl.col("x")),
    "ppf": lambda b: b.ppf(pl.col("q")),
    "isf": lambda b: b.isf(pl.col("q")),
    "mean": lambda b: b.mean(),
    "variance": lambda b: b.variance(),
    "std": lambda b: b.std(),
    "median": lambda b: b.median(),
    "entropy": lambda b: b.entropy(),
    "sample": lambda b: b.sample(seed=0),
    "samples": lambda b: b.samples(size=4, seed=0),
}


@pytest.mark.parametrize("expr_fn", _METHODS.values(), ids=list(_METHODS))
def test_method_raises_on_out_of_range_p_column(expr_fn: Callable[[Binomial], pl.Expr]) -> None:
    df = pl.DataFrame({"n": [10, 10], "p": [0.5, 1.5], "x": [0, 0], "q": [0.5, 0.5]})  # row 1: p out of [0, 1]
    b = Binomial(pl.col("n"), pl.col("p"))
    with pytest.raises(pl.exceptions.ComputeError, match="p must be in"):
        df.select(r=expr_fn(b))


@pytest.mark.parametrize("expr_fn", _METHODS.values(), ids=list(_METHODS))
def test_method_raises_on_out_of_range_p_scalar(expr_fn: Callable[[Binomial], pl.Expr]) -> None:
    df = pl.DataFrame({"x": [0], "q": [0.5]})
    b = Binomial(10, 1.5)
    with pytest.raises(pl.exceptions.ComputeError, match="p must be in"):
        df.select(r=expr_fn(b))


@pytest.mark.parametrize("expr_fn", _METHODS.values(), ids=list(_METHODS))
def test_method_raises_on_negative_n_column(expr_fn: Callable[[Binomial], pl.Expr]) -> None:
    df = pl.DataFrame({"n": [10, -1], "p": [0.5, 0.5], "x": [0, 0], "q": [0.5, 0.5]})  # row 1: n < 0
    b = Binomial(pl.col("n"), pl.col("p"))
    with pytest.raises(pl.exceptions.ComputeError, match="n must be a non-negative integer"):
        df.select(r=expr_fn(b))
