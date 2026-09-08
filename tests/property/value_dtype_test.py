"""Value-keyed methods accept every numeric evaluation-point dtype; non-numeric dtypes fail fast.

The public methods hand the coerced value expression to the Rust plugin as-is, with no `Float64` cast
in Python. The plugin's dtype gate casts every numeric dtype (`Int*`, `UInt*`, `Float*`, `Decimal`) to
`Float64` and refuses everything else with a `ComputeError`, in both positions. Pinned here so a
regression in either direction surfaces:

* a narrow value column, or the integer literal a scalar like `cdf(0)` coerces to, must evaluate
  exactly as its `Float64`-cast equivalent; every grid here is exact in every dtype it is cast to.
* a non-numeric value column (`Boolean`, `String`, temporal, nested) is rejected before any row
  computes: a numeric `String` column must not silently parse, and a `Boolean` one must not compute
  as `0` / `1`, which is what polars' own cast would do with both.

Whether a dtype reaches the plugin at all is `tests/plugin_boundary_dtype_test.py`'s question.
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
from tests._polars_compat import assert_series_equal, available_dtypes
from tests.property._specs import ALL_SPECS

if TYPE_CHECKING:
    from polars_stats.distributions._base import _UnivariateDistribution
    from tests.property._specs import DistSpec

_VALUE_DTYPES = (
    pl.Int64(),
    pl.UInt32(),
    pl.Float32(),
    pl.Decimal(10, 2),
    *available_dtypes("Int128", "UInt128", "Float16"),
)
"""`Int64` and `UInt32` cover the integer contract and `Decimal` is the one numeric dtype that is neither integer nor
float; `Int128`, `UInt128` and `Float16` each need a polars build feature."""

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
    q_grid = [0, 1, None] if dtype.is_integer() else [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, None]
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

    One method suffices: the `as_expr` coercion this scalar routes through, and the plugin gate that
    casts it, are shared by every value-keyed method.
    """
    params = data.draw(spec.params)
    dist = spec.make(params)
    frame = pl.DataFrame({"rows": [0.0, 0.0, 0.0]})
    assert_series_equal(frame.select(r=dist.cdf(1))["r"], frame.select(r=dist.cdf(1.0))["r"], check_exact=True)


_NON_NUMERIC_COLUMNS = (
    pl.Series("x", [True, False]),
    pl.Series("x", ["0.5", "1.0"]),
    pl.Series("x", ["a", "b"], dtype=pl.Categorical),
    pl.Series("x", ["a", "b"], dtype=pl.Enum(["a", "b"])),
    pl.Series("x", [{"a": 1}, {"a": 2}], dtype=pl.Struct({"a": pl.Int64})),
    pl.Series("x", [1, 2], dtype=pl.Int32).cast(pl.Date),
    pl.Series("x", [1, 2], dtype=pl.Int64).cast(pl.Datetime("us")),
    pl.Series("x", [1, 2], dtype=pl.Int64).cast(pl.Duration("us")),
    pl.Series("x", [1, 2], dtype=pl.Int64).cast(pl.Time),
)
"""Every dtype the Rust gate refuses, in either position, where polars' own cast would compute."""


@pytest.mark.parametrize(
    "dist",
    [Normal(mu=0.0, sigma=1.0), Uniform(min=0.0, max=1.0)],
    ids=["normal", "uniform"],
)
@pytest.mark.parametrize("series", _NON_NUMERIC_COLUMNS, ids=lambda s: str(s.dtype))
def test_non_numeric_value_column_raises(dist: _UnivariateDistribution, series: pl.Series) -> None:
    """A non-numeric evaluation point is refused before any row computes."""
    with pytest.raises(pl.exceptions.ComputeError, match="'x' must be a numeric column"):
        series.to_frame().select(dist.cdf(pl.col("x")))


@pytest.mark.parametrize("series", _NON_NUMERIC_COLUMNS, ids=lambda s: str(s.dtype))
@pytest.mark.parametrize("method", ["mean", "sample"], ids=str)
def test_non_numeric_parameter_column_raises_from_the_plugin(series: pl.Series, method: str) -> None:
    """A non-numeric *parameter* raises `ComputeError` from Rust, on a moment and on a sampler alike."""
    dist = Normal(mu="mu", sigma=1.0)
    expr = dist.mean() if method == "mean" else dist.sample(seed=0)
    with pytest.raises(pl.exceptions.ComputeError, match="'mu' must be a numeric column"):
        series.rename("mu").to_frame().select(r=expr)


@pytest.mark.parametrize("dtype", [*available_dtypes("Int128", "UInt128", "Float16"), pl.Decimal(10, 2)], ids=str)
def test_feature_gated_parameter_column_computes(dtype: pl.DataType) -> None:
    """A parameter column in a build-feature-gated dtype reaches Rust with its value intact.

    `cdf`, not a moment: a moment returns the parameter column itself and never reads its value.
    """
    frame = pl.Series("mu", [1, 2], dtype=pl.Int64()).cast(dtype).to_frame()
    narrow = frame.select(r=Normal(mu="mu", sigma=1.0).cdf(0.0))["r"]
    wide = frame.select(r=Normal(mu=pl.col("mu").cast(pl.Float64()), sigma=1.0).cdf(0.0))["r"]
    assert_series_equal(narrow, wide, check_exact=True)
