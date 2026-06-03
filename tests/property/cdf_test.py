from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests._polars_compat import assert_series_equal, linear_space
from tests.property._specs import ALL_SPECS, CONTINUOUS_SPECS

if TYPE_CHECKING:
    from collections.abc import Iterable

    from tests.property._specs import DistSpec

_GRID_SIZE = 64
# cdf/ppf on the normal go through `erf`/`erfc` and a closed-form `inverse_cdf`, so the round-trip
# recovers `q` only to ~1e-9. 1e-6 is the padded bound covering every distribution uniformly.
_ROUNDTRIP_TOL = 1e-6
# Float-noise slack on strict monotonicity: a genuine cdf decrease must still fail.
_MONOTONE_SLACK = 1e-12


def _eval(expr: pl.Expr, xs: Iterable[float]) -> pl.Series:
    return pl.DataFrame({"x": xs}).select(r=expr)["r"]


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.name)
@given(data=st.data())
def test_cdf_bounded_and_monotone(spec: DistSpec, data: st.DataObject) -> None:
    """`0 <= cdf(x) <= 1` and `cdf` is non-decreasing in `x`, across the parameter space."""
    params = data.draw(spec.params)
    dist = spec.make(params)
    lo, hi = spec.eval_range(params)
    xs = linear_space(lo, hi, _GRID_SIZE)

    cdf = _eval(dist.cdf(pl.col("x")), xs)

    assert not cdf.is_nan().any()
    assert cdf.is_between(0.0, 1.0).all()
    # Allow only float noise against strict monotonicity, not a real decrease.
    assert cdf.diff().ge(-_MONOTONE_SLACK).all()


@pytest.mark.parametrize("spec", CONTINUOUS_SPECS, ids=lambda s: s.name)
@given(q=st.floats(min_value=1e-3, max_value=1.0 - 1e-3), data=st.data())
def test_cdf_ppf_round_trip(spec: DistSpec, q: float, data: st.DataObject) -> None:
    """`cdf(ppf(q)) ~= q` for `q` in the open unit interval (continuous distributions)."""
    params = data.draw(spec.params)
    dist = spec.make(params)

    recovered = _eval(dist.cdf(dist.ppf(pl.col("x"))), [q])
    assert_series_equal(recovered, pl.Series([q]), rel_tol=0.0, abs_tol=_ROUNDTRIP_TOL, check_names=False)
