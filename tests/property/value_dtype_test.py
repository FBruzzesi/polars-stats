"""Value-keyed methods accept narrow numeric evaluation points; non-numeric dtypes fail fast.

`propagate_null_and_nan` in `_base.py` applies `is_nan` to the coerced value expression as-is,
with no `Float64` cast. That relies on two polars behaviours, both verified on every supported
version (1.15.0 through current) and pinned here so a regression in either direction surfaces:

* `is_nan` returns `False` for integer dtypes (an integer can never hold a `NaN`), so an
  integer-typed value column, or the integer literal a scalar like `cdf(0)` coerces to,
  flows through the guard and must evaluate exactly as its `Float64`-cast equivalent. The Rust
  plugins cast the evaluation point to `Float64` internally; the closed-form hooks combine it
  under polars supertype rules; both are exact for the grids used here.
* `is_nan` raises `InvalidOperationError` for non-numeric dtypes (`Boolean`, `String`, temporal),
  so an invalid value column is rejected up front, before any hook or plugin sees it. This is the
  strict half of the contract: a numeric `String` column must not silently parse through the
  statrs-backed paths (both a Python-side `cast` and the plugin's internal Rust cast would
  otherwise accept it).

Below the guard, Rust refuses every non-numeric dtype with a `ComputeError` in both positions; parameters
have no Python guard, so that is their whole contract. `Decimal` is numeric to Rust and computes in both.

Whether a dtype reaches either half at all is `tests/plugin_boundary_dtype_test.py`'s question.
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import TYPE_CHECKING

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from polars_stats import Normal, Uniform
from polars_stats.distributions._base import ContinuousDistribution, DiscreteDistribution
from tests._polars_compat import assert_series_equal, available_dtypes
from tests.property._specs import ALL_SPECS

if TYPE_CHECKING:
    from polars_stats.distributions._base import _UnivariateDistribution
    from tests.property._specs import DistSpec

_VALUE_DTYPES = (pl.Int64(), pl.UInt32(), pl.Float32(), *available_dtypes("Int128", "UInt128", "Float16"))
"""`Int64` and `UInt32` cover the contract; `Int128`, `UInt128` and `Float16` each need a polars build feature."""

_MAX_GRID = 16


def _int_grid(lo: float, hi: float, dtype: pl.DataType) -> list[int | None]:
    """Integers spanning `[floor(lo), ceil(hi)]`, thinned to at most ~`_MAX_GRID` points, plus a null probe.

    An unsigned dtype clamps the window at 0 (kept non-empty), since it cannot represent the
    negative part of a signed evaluation range.
    """
    lo_i, hi_i = math.floor(lo), math.ceil(hi)
    if dtype.is_unsigned_integer():
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


@pytest.mark.parametrize("dtype", _VALUE_DTYPES, ids=str)
@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.name)
@given(data=st.data())
def test_narrow_value_column_matches_float64(spec: DistSpec, dtype: pl.DataType, data: st.DataObject) -> None:
    """Every value-keyed method evaluates a narrow value column exactly as its `Float64` cast, nulls included.

    Every grid value is exactly representable in every dtype here, so a mismatch is a misread buffer,
    not a hook computing at the column's width.
    """
    params = data.draw(spec.params)
    dist = spec.make(params)
    lo, hi = spec.eval_range(params)
    values = pl.Series("x", _int_grid(lo, hi, dtype)).cast(dtype).to_frame()
    q_grid = [0, 1, None] if dtype.is_integer() else [0.0, 0.25, 0.5, 0.75, 1.0, None]
    quantiles = pl.Series("q", q_grid).cast(dtype).to_frame()

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
    for frame, narrow_expr, wide_expr in cases:
        assert_series_equal(frame.select(r=narrow_expr)["r"], frame.select(r=wide_expr)["r"], check_exact=True)


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


_REFUSED_VALUES = (
    pl.Series("x", [True, False]),
    pl.Series("x", ["0.5", "1.0"]),
    pl.Series("x", [Decimal("0.50"), Decimal("1.00")], dtype=pl.Decimal(10, 2)),
    pl.Series("x", ["a", "b"], dtype=pl.Categorical),
    pl.Series("x", ["a", "b"], dtype=pl.Enum(["a", "b"])),
    pl.Series("x", [{"a": 1}, {"a": 2}], dtype=pl.Struct({"a": pl.Int64})),
    pl.Series("x", [1, 2], dtype=pl.Int32).cast(pl.Date),
    pl.Series("x", [1, 2], dtype=pl.Int64).cast(pl.Datetime("us")),
    pl.Series("x", [1, 2], dtype=pl.Int64).cast(pl.Duration("us")),
    pl.Series("x", [1, 2], dtype=pl.Int64).cast(pl.Time),
)
"""Every dtype the public guard refuses in the value position. `Decimal` is numeric, but `is_nan` has no `NaN` on it."""


@pytest.mark.parametrize(
    "dist",
    [Normal(mu=0.0, sigma=1.0), Uniform(min=0.0, max=1.0)],
    ids=["normal", "uniform"],
)
@pytest.mark.parametrize("series", _REFUSED_VALUES, ids=lambda s: str(s.dtype))
def test_non_numeric_value_column_raises(dist: _UnivariateDistribution, series: pl.Series) -> None:
    """A non-numeric value column is rejected up front, plugin-backed or closed-form alike.

    Only the exception type is pinned, not the message: whether the guard's `is_nan` or a hook
    operation (e.g. `Uniform`'s division on a `String`) resolves first is a polars
    schema-resolution ordering detail. The contract is that the query errors instead of silently
    computing (`InvalidOperationError` on every supported polars for both operators).
    """
    with pytest.raises(pl.exceptions.InvalidOperationError):
        series.to_frame().select(dist.cdf(pl.col("x")))


_REFUSED_PARAMETERS = tuple(series.rename("mu") for series in _REFUSED_VALUES if not series.dtype.is_decimal())
"""Every non-numeric dtype; Rust refuses each as a parameter where polars' own cast would compute."""


@pytest.mark.parametrize("series", _REFUSED_PARAMETERS, ids=lambda s: str(s.dtype))
@pytest.mark.parametrize("method", ["mean", "sample"], ids=str)
def test_refused_parameter_column_raises_from_the_plugin(series: pl.Series, method: str) -> None:
    """A refused *parameter* raises `ComputeError` from Rust: no Python-side guard sees a parameter column."""
    dist = Normal(mu="mu", sigma=1.0)
    expr = dist.mean() if method == "mean" else dist.sample(seed=0)
    with pytest.raises(pl.exceptions.ComputeError, match="'mu' must be a numeric column"):
        series.to_frame().select(r=expr)


@pytest.mark.parametrize(
    "dist",
    [Normal(mu=0.0, sigma=1.0), Uniform(min=0.0, max=1.0)],
    ids=["normal", "uniform"],
)
@pytest.mark.parametrize("series", _REFUSED_PARAMETERS, ids=lambda s: str(s.dtype))
def test_non_numeric_value_column_raises_from_the_plugin(dist: _UnivariateDistribution, series: pl.Series) -> None:
    """Below the public guard, the funnel refuses a non-numeric evaluation point: nothing parses, nothing computes."""
    with pytest.raises(pl.exceptions.ComputeError, match="'x' must be a numeric column"):
        series.rename("x").to_frame().select(dist._cdf(pl.col("x")))


@pytest.mark.parametrize(
    "dist",
    [Normal(mu=0.0, sigma=1.0), Uniform(min=0.0, max=1.0)],
    ids=["normal", "uniform"],
)
def test_decimal_value_column_computes_through_the_hook(dist: _UnivariateDistribution) -> None:
    """A `Decimal` evaluation point is numeric to the funnel and evaluates as its `Float64` cast."""
    frame = pl.Series("x", [Decimal("0.50"), Decimal("0.25"), None], dtype=pl.Decimal(10, 2)).to_frame()
    narrow = frame.select(r=dist._cdf(pl.col("x")))["r"]
    wide = frame.select(r=dist._cdf(pl.col("x").cast(pl.Float64())))["r"]
    assert_series_equal(narrow, wide, check_exact=True)


@pytest.mark.parametrize("dtype", [*available_dtypes("Int128", "UInt128", "Float16"), pl.Decimal(10, 2)], ids=str)
def test_feature_gated_parameter_column_computes(dtype: pl.DataType) -> None:
    """A parameter column in a build-feature-gated dtype reaches Rust with its value intact.

    `cdf`, not a moment: a moment returns the parameter column itself and never reads its value.
    """
    frame = pl.Series("mu", [1, 2], dtype=pl.Int64()).cast(dtype).to_frame()
    narrow = frame.select(r=Normal(mu="mu", sigma=1.0).cdf(0.0))["r"]
    wide = frame.select(r=Normal(mu=pl.col("mu").cast(pl.Float64()), sigma=1.0).cdf(0.0))["r"]
    assert_series_equal(narrow, wide, check_exact=True)
