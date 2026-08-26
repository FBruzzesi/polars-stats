"""Value-keyed methods accept integer-typed evaluation points; non-numeric dtypes fail fast.

`propagate_null_and_nan` in `_base.py` applies `is_nan` to the coerced value expression as-is,
with no `Float64` cast. That relies on two polars behaviours, both verified on every supported
version (1.15.0 through current) and pinned here so a regression in either direction surfaces:

* `is_nan` returns `False` for integer dtypes (an integer can never hold a `NaN`), so an
  integer-typed value column, or the integer literal a scalar like `cdf(0)` coerces to,
  flows through the guard and must evaluate exactly as its `Float64`-cast equivalent. The Rust
  plugins cast the evaluation point to `Float64` internally; the closed-form hooks combine it
  under polars supertype rules; both are exact for the integer grids used here.
* `is_nan` raises `InvalidOperationError` for non-numeric dtypes (`Boolean`, `String`, temporal),
  so an invalid value column is rejected up front, before any hook or plugin sees it. This is the
  strict half of the contract: a numeric `String` column must not silently parse through the
  statrs-backed paths (both a Python-side `cast` and the plugin's internal Rust cast would
  otherwise accept it).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from polars_stats import Normal, Uniform
from polars_stats.distributions._base import ContinuousDistribution, DiscreteDistribution
from tests._polars_compat import assert_series_equal
from tests.property._specs import ALL_SPECS

if TYPE_CHECKING:
    from polars_stats.distributions._base import _UnivariateDistribution
    from tests.property._specs import DistSpec

_INT_DTYPES = (pl.Int64(), pl.UInt32())
"""One signed and one unsigned integer dtype; the guard and the hooks are dtype-generic past that."""

_MAX_GRID = 16


def _int_grid(lo: float, hi: float, dtype: pl.DataType) -> list[int | None]:
    """Integers spanning `[floor(lo), ceil(hi)]`, thinned to at most ~`_MAX_GRID` points, plus a null probe.

    An unsigned dtype clamps the window at 0 (kept non-empty), since it cannot represent the
    negative part of a signed evaluation range.
    """
    lo_i, hi_i = math.floor(lo), math.ceil(hi)
    if dtype in (pl.UInt8(), pl.UInt16(), pl.UInt32(), pl.UInt64()):
        lo_i = max(lo_i, 0)
        hi_i = max(hi_i, lo_i)
    step = max(1, (hi_i - lo_i) // _MAX_GRID)
    return [*range(lo_i, hi_i + 1, step), None]


def _log_density(dist: _UnivariateDistribution, value: pl.Expr) -> pl.Expr:
    """`log_pdf` / `log_pmf` by family; the method lives on the family subclass, hence the narrowing."""
    if isinstance(dist, ContinuousDistribution):
        return dist.log_pdf(value)
    if isinstance(dist, DiscreteDistribution):
        return dist.log_pmf(value)
    msg = f"unsupported distribution family: {type(dist)}"  # pragma: no cover
    raise TypeError(msg)  # pragma: no cover


@pytest.mark.parametrize("dtype", _INT_DTYPES, ids=str)
@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.name)
@given(data=st.data())
def test_integer_value_column_matches_float64(spec: DistSpec, dtype: pl.DataType, data: st.DataObject) -> None:
    """Every value-keyed method evaluates an integer value column exactly as its `Float64` cast.

    A null in the integer column must also propagate identically (the guard's null branch is
    dtype-agnostic, but this pins it on the integer path specifically).
    """
    params = data.draw(spec.params)
    dist = spec.make(params)
    lo, hi = spec.eval_range(params)
    values = pl.DataFrame({"x": _int_grid(lo, hi, dtype)}, schema={"x": dtype})
    quantiles = pl.DataFrame({"q": [0, 1, None]}, schema={"q": dtype})

    x, q = pl.col("x"), pl.col("q")
    cases = [
        (values, spec.density(dist, x), spec.density(dist, x.cast(pl.Float64()))),
        (values, _log_density(dist, x), _log_density(dist, x.cast(pl.Float64()))),
        (values, dist.cdf(x), dist.cdf(x.cast(pl.Float64()))),
        (values, dist.log_cdf(x), dist.log_cdf(x.cast(pl.Float64()))),
        (values, dist.sf(x), dist.sf(x.cast(pl.Float64()))),
        (values, dist.log_sf(x), dist.log_sf(x.cast(pl.Float64()))),
        (quantiles, dist.ppf(q), dist.ppf(q.cast(pl.Float64()))),
        (quantiles, dist.isf(q), dist.isf(q.cast(pl.Float64()))),
    ]
    for frame, int_expr, float_expr in cases:
        as_int = frame.select(r=int_expr)["r"]
        as_float = frame.select(r=float_expr)["r"]
        assert_series_equal(as_int, as_float, check_exact=True)


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.name)
@given(data=st.data())
def test_integer_scalar_value_matches_float_scalar(spec: DistSpec, data: st.DataObject) -> None:
    """`cdf(1)` (coerced by `as_expr` to a length-1 integer literal) equals `cdf(1.0)`.

    One method suffices: the `as_expr` coercion and the `propagate_null_and_nan` guard this scalar
    routes through are shared by every value-keyed method.
    """
    params = data.draw(spec.params)
    dist = spec.make(params)
    frame = pl.DataFrame({"rows": [0.0, 0.0, 0.0]})
    assert_series_equal(frame.select(r=dist.cdf(1))["r"], frame.select(r=dist.cdf(1.0))["r"], check_exact=True)


@pytest.mark.parametrize(
    "dist",
    [Normal(mu=0.0, sigma=1.0), Uniform(min=0.0, max=1.0)],
    ids=["normal", "uniform"],
)
@pytest.mark.parametrize(
    "series",
    [pl.Series("x", [True, False]), pl.Series("x", ["0.5", "1.0"])],
    ids=["boolean", "string"],
)
def test_non_numeric_value_column_raises(dist: _UnivariateDistribution, series: pl.Series) -> None:
    """A `Boolean` or `String` value column is rejected up front, plugin-backed or closed-form alike.

    Only the exception type is pinned, not the message: whether the guard's `is_nan` or a hook
    operation (e.g. `Uniform`'s division on a `String`) resolves first is a polars
    schema-resolution ordering detail. The contract is that the query errors instead of silently
    computing (`InvalidOperationError` on every supported polars for both operators).
    """
    with pytest.raises(pl.exceptions.InvalidOperationError):
        series.to_frame().select(dist.cdf(pl.col("x")))
