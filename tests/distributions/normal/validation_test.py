from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from polars_stats import Normal

if TYPE_CHECKING:
    from collections.abc import Callable

# Every public method must report an invalid scale (`std_dev <= 0`) as a ComputeError, not silently
# compute garbage or null the row. Each method builds the distribution in Rust via `statrs`, which is
# what enforces this consistently across the surface, and identically for scalar and column inputs.
_METHODS: dict[str, Callable[[Normal], pl.Expr]] = {
    "pdf": lambda n: n.pdf(pl.col("x")),
    "log_pdf": lambda n: n.log_pdf(pl.col("x")),
    "cdf": lambda n: n.cdf(pl.col("x")),
    "log_cdf": lambda n: n.log_cdf(pl.col("x")),
    "sf": lambda n: n.sf(pl.col("x")),
    "log_sf": lambda n: n.log_sf(pl.col("x")),
    "ppf": lambda n: n.ppf(pl.col("q")),
    "isf": lambda n: n.isf(pl.col("q")),
    "mean": lambda n: n.mean(),
    "variance": lambda n: n.variance(),
    "std": lambda n: n.std(),
    "median": lambda n: n.median(),
    "entropy": lambda n: n.entropy(),
    "sample": lambda n: n.sample(seed=0),
    "samples": lambda n: n.samples(size=4, seed=0),
}


@pytest.mark.parametrize("expr_fn", _METHODS.values(), ids=list(_METHODS))
def test_method_raises_on_non_positive_std_column(expr_fn: Callable[[Normal], pl.Expr]) -> None:
    df = pl.DataFrame({"mu": [0.0, 1.0], "sigma": [1.0, -2.0], "x": [0.5, 0.5], "q": [0.5, 0.5]})
    n = Normal(mean=pl.col("mu"), std_dev=pl.col("sigma"))  # row 1: sigma = -2.0
    with pytest.raises(pl.exceptions.ComputeError, match="std_dev must be finite and strictly positive"):
        df.select(r=expr_fn(n))


@pytest.mark.parametrize("expr_fn", _METHODS.values(), ids=list(_METHODS))
def test_method_raises_on_non_positive_std_scalar(expr_fn: Callable[[Normal], pl.Expr]) -> None:
    df = pl.DataFrame({"x": [0.5], "q": [0.5]})
    n = Normal(mean=0.0, std_dev=-1.0)
    with pytest.raises(pl.exceptions.ComputeError, match="std_dev must be finite and strictly positive"):
        df.select(r=expr_fn(n))
