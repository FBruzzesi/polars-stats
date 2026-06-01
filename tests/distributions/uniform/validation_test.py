from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from polars_stats import Uniform

if TYPE_CHECKING:
    from collections.abc import Callable

# Every public method must report an invalid parameterisation (`max <= min`) as a ComputeError, not
# silently compute garbage. The deterministic methods all derive from the Rust-validated `range`; the
# samplers validate in their own plugin.
_METHODS: dict[str, Callable[[Uniform], pl.Expr]] = {
    "pdf": lambda u: u.pdf(pl.col("x")),
    "log_pdf": lambda u: u.log_pdf(pl.col("x")),
    "cdf": lambda u: u.cdf(pl.col("x")),
    "log_cdf": lambda u: u.log_cdf(pl.col("x")),
    "sf": lambda u: u.sf(pl.col("x")),
    "log_sf": lambda u: u.log_sf(pl.col("x")),
    "ppf": lambda u: u.ppf(pl.col("q")),
    "isf": lambda u: u.isf(pl.col("q")),
    "mean": lambda u: u.mean(),
    "variance": lambda u: u.variance(),
    "std": lambda u: u.std(),
    "median": lambda u: u.median(),
    "entropy": lambda u: u.entropy(),
    "sample": lambda u: u.sample(seed=0),
    "samples": lambda u: u.samples(size=4, seed=0),
    "range": lambda u: u.range,
}


@pytest.mark.parametrize("expr_fn", _METHODS.values(), ids=list(_METHODS))
def test_method_raises_on_max_le_min_column(expr_fn: Callable[[Uniform], pl.Expr]) -> None:
    df = pl.DataFrame({"lo": [0.0, 5.0], "hi": [1.0, 2.0], "x": [0.5, 0.5], "q": [0.5, 0.5]})
    u = Uniform(min=pl.col("lo"), max=pl.col("hi"))  # row 1: hi (2.0) <= lo (5.0)
    with pytest.raises(pl.exceptions.ComputeError, match="max must be strictly greater than min"):
        df.select(r=expr_fn(u))


@pytest.mark.parametrize("expr_fn", _METHODS.values(), ids=list(_METHODS))
def test_method_raises_on_max_le_min_scalar(expr_fn: Callable[[Uniform], pl.Expr]) -> None:
    df = pl.DataFrame({"x": [0.5], "q": [0.5]})
    u = Uniform(min=5.0, max=2.0)
    with pytest.raises(pl.exceptions.ComputeError, match="max must be strictly greater than min"):
        df.select(r=expr_fn(u))
