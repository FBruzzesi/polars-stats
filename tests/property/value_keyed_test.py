"""Bit-equality of the three parameter spellings on every value-keyed method.

Constant scalar parameters route the value-keyed methods (density, log-density, cdf, log-cdf, sf,
log-sf, ppf, isf) through a dedicated ``<name>_<method>_scalar`` plugin: parameters are validated once and
passed as kwargs, with only the value column crossing FFI. Expression parameters take the general per-row
plugin, which validates and builds once per call when every parameter is length 1 (a `pl.lit`, an aggregate)
and once per row otherwise. In Rust all three spellings call the same named per-method body, so for any
parameterisation they must agree bit for bit, including null propagation and ppf's null-outside-``[0, 1]``
contract. A divergence (e.g. a parameter-order swap in a scalar kwargs struct) must fail here.

The comparison is bit-exact for every spec.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from polars_stats.distributions._base import ContinuousDistribution, DiscreteDistribution
from tests._polars_compat import assert_series_equal, linear_space
from tests.property._specs import ALL_SPECS

if TYPE_CHECKING:
    from collections.abc import Callable

    from polars_stats.distributions._base import _UnivariateDistribution
    from tests.property._specs import DistSpec

_GRID_SIZE = 64

_PPF_EDGE_QUANTILES = (-0.5, 0.0, 1.0, 1.5, None, float("nan"))
"""Out-of-range quantiles exercise ppf's null path; the closed endpoints exercise its boundary mapping.

A null or NaN in the value column must propagate on both paths.
"""


def _log_density(dist: _UnivariateDistribution, value: pl.Expr) -> pl.Expr:
    """`log_pdf` / `log_pmf` by family; the method lives on the family subclass, hence the narrowing."""
    if isinstance(dist, ContinuousDistribution):
        return dist.log_pdf(value)
    if isinstance(dist, DiscreteDistribution):
        return dist.log_pmf(value)
    msg = f"unsupported distribution family: {type(dist)}"  # pragma: no cover
    raise TypeError(msg)  # pragma: no cover


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.name)
@given(data=st.data())
def test_value_keyed_scalar_fast_path_matches_per_row(spec: DistSpec, data: st.DataObject) -> None:
    """Constant scalar parameters, length-1 literals and the equivalent per-row columns evaluate identically."""
    params = data.draw(spec.params)
    scalar = spec.make(params)
    per_row = spec.make_columns(params)
    literal = spec.make_literals(params)

    lo, hi = spec.eval_range(params)
    values = pl.DataFrame({"x": [*linear_space(lo, hi, _GRID_SIZE), None, float("nan")]}, schema={"x": pl.Float64})
    quantiles = pl.DataFrame(
        {"q": [*linear_space(1e-3, 1.0 - 1e-3, _GRID_SIZE), *_PPF_EDGE_QUANTILES]},
        schema={"q": pl.Float64},
    )

    x, q = pl.col("x"), pl.col("q")
    cases: list[tuple[pl.DataFrame, Callable[[_UnivariateDistribution], pl.Expr]]] = [
        (values, lambda d: spec.density(d, x)),
        (values, lambda d: _log_density(d, x)),
        (values, lambda d: d.cdf(x)),
        (values, lambda d: d.log_cdf(x)),
        (values, lambda d: d.sf(x)),
        (values, lambda d: d.log_sf(x)),
        (quantiles, lambda d: d.ppf(q)),
        (quantiles, lambda d: d.isf(q)),
    ]
    for frame, method in cases:
        fast = frame.select(r=method(scalar))["r"]
        for spelling in (per_row, literal):
            assert_series_equal(fast, frame.select(r=method(spelling))["r"], check_exact=True)
