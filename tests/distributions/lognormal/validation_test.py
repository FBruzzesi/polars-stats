from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from polars_stats import LogNormal

if TYPE_CHECKING:
    from collections.abc import Callable

# Every public method must report an invalid scale (`sigma <= 0`) as a ComputeError, not silently
# compute garbage or null the row. Each method builds the distribution in Rust via `statrs`, which is
# what enforces this consistently across the surface, and identically for scalar and column inputs.
_METHODS: dict[str, Callable[[LogNormal], pl.Expr]] = {
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
def test_method_raises_on_non_positive_sigma_column(expr_fn: Callable[[LogNormal], pl.Expr]) -> None:
    df = pl.DataFrame({"mu": [0.0, 1.0], "sigma": [1.0, -2.0], "x": [1.5, 1.5], "q": [0.5, 0.5]})
    d = LogNormal(mu=pl.col("mu"), sigma=pl.col("sigma"))  # row 1: sigma = -2.0
    with pytest.raises(pl.exceptions.ComputeError, match="sigma must be finite and strictly positive"):
        df.select(r=expr_fn(d))


@pytest.mark.parametrize("expr_fn", _METHODS.values(), ids=list(_METHODS))
def test_method_raises_on_non_positive_sigma_scalar(expr_fn: Callable[[LogNormal], pl.Expr]) -> None:
    df = pl.DataFrame({"x": [1.5], "q": [0.5]})
    d = LogNormal(mu=0.0, sigma=-1.0)
    with pytest.raises(pl.exceptions.ComputeError, match="sigma must be finite and strictly positive"):
        df.select(r=expr_fn(d))
