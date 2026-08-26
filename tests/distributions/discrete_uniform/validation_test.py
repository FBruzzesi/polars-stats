from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from polars_stats import DiscreteUniform

if TYPE_CHECKING:
    from collections.abc import Callable
# Every public method must report `max < min` (or an overflowing width) as a ComputeError, not
# silently compute. All of them derive from the Rust-validated support count; the samplers validate
# in their own plugin.
_METHODS: dict[str, Callable[[DiscreteUniform], pl.Expr]] = {
    "pmf": lambda d: d.pmf(pl.col("x")),
    "log_pmf": lambda d: d.log_pmf(pl.col("x")),
    "cdf": lambda d: d.cdf(pl.col("x")),
    "log_cdf": lambda d: d.log_cdf(pl.col("x")),
    "sf": lambda d: d.sf(pl.col("x")),
    "log_sf": lambda d: d.log_sf(pl.col("x")),
    "ppf": lambda d: d.ppf(0.5),
    "isf": lambda d: d.isf(0.5),
    "mean": lambda d: d.mean(),
    "variance": lambda d: d.variance(),
    "std": lambda d: d.std(),
    "median": lambda d: d.median(),
    "entropy": lambda d: d.entropy(),
    "sample": lambda d: d.sample(seed=0),
    "samples": lambda d: d.samples(size=4, seed=0),
}


@pytest.mark.parametrize("lo_hi", [(6, 1), (-2, -7)])
@pytest.mark.parametrize("expr_fn", _METHODS.values(), ids=list(_METHODS))
def test_method_raises_on_inverted_bounds_column(
    lo_hi: tuple[int, int], expr_fn: Callable[[DiscreteUniform], pl.Expr]
) -> None:
    lo, hi = lo_hi
    df = pl.DataFrame({"lo": [0, lo], "hi": [5, hi], "x": [2.0, 2.0]})
    d = DiscreteUniform(min=pl.col("lo"), max=pl.col("hi"))
    with pytest.raises(pl.exceptions.ComputeError, match="max must be greater than or equal to min"):
        df.select(r=expr_fn(d))


@pytest.mark.parametrize("expr_fn", _METHODS.values(), ids=list(_METHODS))
def test_method_raises_on_inverted_bounds_scalar(expr_fn: Callable[[DiscreteUniform], pl.Expr]) -> None:
    df = pl.DataFrame({"x": [2.0]})
    d = DiscreteUniform(min=6, max=1)
    with pytest.raises(pl.exceptions.ComputeError, match="max must be greater than or equal to min"):
        df.select(r=expr_fn(d))


def test_float_bound_column_is_refused() -> None:
    # The dtype rule is column-level and judged before values: a float bound column is refused even
    # though casting it would be lossless here, because in general it would silently truncate.
    df = pl.DataFrame({"lo": [0.0, 1.0], "hi": [5.0, 6.0]})
    d = DiscreteUniform(min=pl.col("lo"), max=pl.col("hi"))
    with pytest.raises(pl.exceptions.ComputeError, match="bounds must be integer columns"):
        df.select(r=d.mean())


def test_support_width_overflowing_int64_raises() -> None:
    df = pl.DataFrame({"x": [0.0]})
    d = DiscreteUniform(min=-(2**63), max=2**63 - 1)
    with pytest.raises(pl.exceptions.ComputeError, match="support width"):
        df.select(r=d.mean())


def test_min_equal_max_point_mass_is_valid(unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(pmf=DiscreteUniform(min=3, max=3).pmf(3.0), mean=DiscreteUniform(min=3, max=3).mean())
    assert result["pmf"][0] == pytest.approx(1.0)
    assert result["mean"][0] == pytest.approx(3.0)
