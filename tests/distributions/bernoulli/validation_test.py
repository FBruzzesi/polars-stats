from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from polars_stats import Bernoulli

if TYPE_CHECKING:
    from collections.abc import Callable

# Every public method must report an out-of-range `p` as a ComputeError, not silently compute a
# negative probability. The deterministic methods all derive from the Rust-validated `_checked_p`;
# the samplers validate in their own plugin.
_METHODS: dict[str, Callable[[Bernoulli], pl.Expr]] = {
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
def test_method_raises_on_out_of_range_p_column(expr_fn: Callable[[Bernoulli], pl.Expr]) -> None:
    df = pl.DataFrame({"p": [0.5, 1.5], "x": [0, 0], "q": [0.5, 0.5]})  # row 1: p (1.5) out of [0, 1]
    b = Bernoulli(p=pl.col("p"))
    with pytest.raises(pl.exceptions.ComputeError, match="p must be in"):
        df.select(r=expr_fn(b))


@pytest.mark.parametrize("expr_fn", _METHODS.values(), ids=list(_METHODS))
def test_method_raises_on_out_of_range_p_scalar(expr_fn: Callable[[Bernoulli], pl.Expr]) -> None:
    df = pl.DataFrame({"x": [0], "q": [0.5]})
    b = Bernoulli(p=1.5)
    with pytest.raises(pl.exceptions.ComputeError, match="p must be in"):
        df.select(r=expr_fn(b))
