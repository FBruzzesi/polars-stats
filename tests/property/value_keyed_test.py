"""Bit-equality of the constant-parameter value-keyed fast path against the per-row path.

Constant scalar parameters route the value-keyed methods (density, log-density, cdf, log-cdf, sf,
log-sf, ppf, isf) through a dedicated ``<name>_<method>_scalar`` plugin: parameters are validated once and
passed as kwargs, with only the value column crossing FFI. Column parameters take the general per-row plugin.
In Rust both plugins call the same named per-method body, so for any parameterisation the two paths
must agree bit for bit, including null propagation and ppf's null-outside-``[0, 1]`` contract. A
divergence (e.g. a parameter-order swap in a scalar kwargs struct) must fail here.

Distributions without a Rust value-keyed plugin (bernoulli, uniform) evaluate the same Polars
expressions on both paths, so they pass trivially; they stay in the sweep to keep the contract
uniform across ``ALL_SPECS``.
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
    """Constant scalar parameters and the equivalent per-row columns evaluate identically."""
    params = data.draw(spec.params)
    scalar = spec.make(params)
    per_row = spec.make_columns(params)

    lo, hi = spec.eval_range(params)
    values = pl.DataFrame({"x": [*linear_space(lo, hi, _GRID_SIZE), None, float("nan")]}, schema={"x": pl.Float64})
    quantiles = pl.DataFrame(
        {"q": [*linear_space(1e-3, 1.0 - 1e-3, _GRID_SIZE), *_PPF_EDGE_QUANTILES]},
        schema={"q": pl.Float64},
    )

    x, q = pl.col("x"), pl.col("q")
    cases = [
        (values, spec.density(scalar, x), spec.density(per_row, x)),
        (values, _log_density(scalar, x), _log_density(per_row, x)),
        (values, scalar.cdf(x), per_row.cdf(x)),
        (values, scalar.log_cdf(x), per_row.log_cdf(x)),
        (values, scalar.sf(x), per_row.sf(x)),
        (values, scalar.log_sf(x), per_row.log_sf(x)),
        (quantiles, scalar.ppf(q), per_row.ppf(q)),
        (quantiles, scalar.isf(q), per_row.isf(q)),
    ]
    for frame, fast_expr, per_row_expr in cases:
        fast = frame.select(r=fast_expr)["r"]
        slow = frame.select(r=per_row_expr)["r"]
        assert_series_equal(fast, slow, check_exact=True)
