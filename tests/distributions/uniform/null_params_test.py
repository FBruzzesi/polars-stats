from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from polars_stats import Uniform

if TYPE_CHECKING:
    from collections.abc import Callable

# Every method must propagate a null bound to a null result. The value-keyed methods are evaluated at
# an on-support point, where both bounds enter the formula, so the null flows through whichever one is
# missing. The points where only one bound decides the answer are covered below.
_METHODS: dict[str, Callable[[Uniform], pl.Expr]] = {
    "pdf": lambda u: u.pdf(pl.lit(0.5)),
    "log_pdf": lambda u: u.log_pdf(pl.lit(0.5)),
    "cdf": lambda u: u.cdf(pl.lit(0.5)),
    "log_cdf": lambda u: u.log_cdf(pl.lit(0.5)),
    "sf": lambda u: u.sf(pl.lit(0.5)),
    "log_sf": lambda u: u.log_sf(pl.lit(0.5)),
    "ppf": lambda u: u.ppf(pl.lit(0.5)),
    "isf": lambda u: u.isf(pl.lit(0.5)),
    "mean": lambda u: u.mean(),
    "variance": lambda u: u.variance(),
    "std": lambda u: u.std(),
    "median": lambda u: u.median(),
    "entropy": lambda u: u.entropy(),
    "sample": lambda u: u.sample(seed=0),
    "samples": lambda u: u.samples(size=2, seed=0),
}


def _column_bounds() -> Uniform:
    return Uniform(min=pl.col("lo"), max=pl.col("hi"))


@pytest.mark.parametrize("expr_fn", _METHODS.values(), ids=list(_METHODS))
def test_method_propagates_null_in_min(expr_fn: Callable[[Uniform], pl.Expr]) -> None:
    df = pl.DataFrame({"lo": [0.0, None, 0.0], "hi": [1.0, 1.0, 1.0]}, schema={"lo": pl.Float64, "hi": pl.Float64})
    result = df.select(r=expr_fn(_column_bounds()))["r"]
    assert result.is_null().to_list() == [False, True, False]


@pytest.mark.parametrize("expr_fn", _METHODS.values(), ids=list(_METHODS))
def test_method_propagates_null_in_max(expr_fn: Callable[[Uniform], pl.Expr]) -> None:
    # `max` is the *second* plugin input, so a guard written against the first only would pass the
    # test above and fail here.
    df = pl.DataFrame({"lo": [0.0, 0.0, 0.0], "hi": [1.0, None, 1.0]}, schema={"lo": pl.Float64, "hi": pl.Float64})
    result = df.select(r=expr_fn(_column_bounds()))["r"]
    assert result.is_null().to_list() == [False, True, False]


def test_density_nulls_where_the_null_bound_decides_the_support() -> None:
    # `is_between` yields null on a null bound and `pl.when(null)` takes the `otherwise`, so without an
    # explicit null branch the density reports a confident `0.0` (`-inf` for the log) for a row whose
    # support is unknown. `cdf` and `sf` need no such branch, because their `otherwise` is the
    # arithmetic and nulls on its own.
    df = pl.DataFrame({"lo": [0.0, None], "hi": [1.0, 1.0]}, schema={"lo": pl.Float64, "hi": pl.Float64})
    dist = _column_bounds()

    on_support = df.select(pdf=dist.pdf(pl.lit(0.5)), log_pdf=dist.log_pdf(pl.lit(0.5)))
    below = df.select(pdf=dist.pdf(pl.lit(-1.0)), log_pdf=dist.log_pdf(pl.lit(-1.0)))

    assert on_support["pdf"].to_list() == [1.0, None]
    assert on_support["log_pdf"].to_list() == [0.0, None]
    assert below["pdf"].to_list() == [0.0, None]
    assert below["log_pdf"].to_list() == [float("-inf"), None]


def test_density_stays_confident_when_the_known_bound_settles_it() -> None:
    # A null bound does not null the row where the *other* bound already places the point outside the
    # support. `is_between` is `False` there rather than null (Kleene `False & null` is `False`), and
    # the density is `0` for every admissible value of the missing bound, so a guard that nulled these
    # rows would be too blunt.
    schema = {"lo": pl.Float64, "hi": pl.Float64}
    null_min = pl.DataFrame({"lo": [None], "hi": [1.0]}, schema=schema)
    null_max = pl.DataFrame({"lo": [0.0], "hi": [None]}, schema=schema)
    dist = _column_bounds()

    above_known_max = null_min.select(pdf=dist.pdf(pl.lit(2.0)), log_pdf=dist.log_pdf(pl.lit(2.0)))
    below_known_min = null_max.select(pdf=dist.pdf(pl.lit(-1.0)), log_pdf=dist.log_pdf(pl.lit(-1.0)))

    for got in (above_known_max, below_known_min):
        assert got["pdf"].to_list() == [0.0]
        assert got["log_pdf"].to_list() == [float("-inf")]
