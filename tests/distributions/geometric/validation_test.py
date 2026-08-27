from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from polars_stats import Geometric

if TYPE_CHECKING:
    from collections.abc import Callable

# Every public method must report an out-of-range `p` as a ComputeError, not silently compute.
# The support is `0 < p <= 1`, so unlike Bernoulli the degenerate `p = 0` is invalid too. The
# deterministic methods all derive from the Rust-validated `_checked_p`; the samplers validate in
# their own plugin.
_METHODS: dict[str, Callable[[Geometric], pl.Expr]] = {
    "pmf": lambda g: g.pmf(pl.col("k")),
    "log_pmf": lambda g: g.log_pmf(pl.col("k")),
    "cdf": lambda g: g.cdf(pl.col("k")),
    "log_cdf": lambda g: g.log_cdf(pl.col("k")),
    "sf": lambda g: g.sf(pl.col("k")),
    "log_sf": lambda g: g.log_sf(pl.col("k")),
    "ppf": lambda g: g.ppf(pl.col("q")),
    "isf": lambda g: g.isf(pl.col("q")),
    "mean": lambda g: g.mean(),
    "variance": lambda g: g.variance(),
    "std": lambda g: g.std(),
    "median": lambda g: g.median(),
    "entropy": lambda g: g.entropy(),
    "sample": lambda g: g.sample(seed=0),
    "samples": lambda g: g.samples(size=4, seed=0),
}


@pytest.mark.parametrize("bad_p", [1.5, 0.0])
@pytest.mark.parametrize("expr_fn", _METHODS.values(), ids=list(_METHODS))
def test_method_raises_on_out_of_range_p_column(bad_p: float, expr_fn: Callable[[Geometric], pl.Expr]) -> None:
    df = pl.DataFrame({"p": [0.5, bad_p], "k": [1, 1], "q": [0.5, 0.5]})
    g = Geometric(p=pl.col("p"))
    with pytest.raises(pl.exceptions.ComputeError, match="p must be in"):
        df.select(r=expr_fn(g))


@pytest.mark.parametrize("bad_p", [1.5, 0.0])
@pytest.mark.parametrize("expr_fn", _METHODS.values(), ids=list(_METHODS))
def test_method_raises_on_out_of_range_p_scalar(bad_p: float, expr_fn: Callable[[Geometric], pl.Expr]) -> None:
    df = pl.DataFrame({"k": [1], "q": [0.5]})
    g = Geometric(p=bad_p)
    with pytest.raises(pl.exceptions.ComputeError, match="p must be in"):
        df.select(r=expr_fn(g))
