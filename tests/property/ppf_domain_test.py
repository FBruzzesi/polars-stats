"""The `ppf` domain contract, across every distribution and both parameter regimes.

`_UnivariateDistribution.ppf` documents `quantile` outside `[0, 1]` as yielding **null**. That used
to be a disclaimer ("implementation-defined and should not be relied on") while every implementation
in fact nulled consistently, so users would have discovered and relied on the de-facto behaviour
while the docstring told them not to. It is now a guarantee, and a guarantee needs a test that
covers every distribution rather than the handful whose per-method files happened to assert it.

Three things are pinned here, all of them boundary behaviour a parameter sweep alone would miss:

* out of range yields null, on both sides and out to the infinities;
* the closed endpoints `0` and `1` are *in* range and never null, though the value they map to (a
  finite support bound, or an infinite tail) is the distribution's own business;
* `isf` honours both. It used to inherit them by definition (`ppf(1 - quantile)`); five
  distributions now implement it independently, so the contract has to be asserted rather than
  deduced, and the endpoints map in the *opposite* order (`isf(0)` is `ppf(1)`).

`ppf` also propagates a null input and maps `NaN` to `NaN`; those belong to the shared value-keyed
contract and are covered by `value_keyed_test.py` and `plugin_nan_test.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests.property._specs import ALL_SPECS

if TYPE_CHECKING:
    from polars_stats.distributions._base import _UnivariateDistribution
    from tests.property._specs import DistSpec

_OUT_OF_RANGE = [-float("inf"), -1e300, -1.5, -0.5, -1e-12, 1.0 + 1e-12, 1.5, 1e300, float("inf")]
"""Just outside on both sides, and far outside. `-1e-12` and `1 + 1e-12` are the interesting ones:
they are the nearest a real quantile column gets to the boundary after arithmetic."""

_ENDPOINTS = [0.0, 1.0]
"""In range by contract. The mapped value may be infinite, so only nullity is asserted."""


def _both_regimes(spec: DistSpec, params: tuple[float, ...]) -> list[tuple[str, _UnivariateDistribution]]:
    """The same parameterisation as constants (fast path) and as column exprs (per-row path)."""
    return [("scalar", spec.make(params)), ("columns", spec.make_columns(params))]


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.name)
@given(data=st.data())
def test_ppf_and_isf_are_null_outside_the_unit_interval(spec: DistSpec, data: st.DataObject) -> None:
    """`q` outside `[0, 1]` yields null, for every distribution and both parameter regimes."""
    params = data.draw(spec.params)
    frame = pl.DataFrame({"q": _OUT_OF_RANGE}, schema={"q": pl.Float64})

    for regime, dist in _both_regimes(spec, params):
        for name, result in (("ppf", dist.ppf(pl.col("q"))), ("isf", dist.isf(pl.col("q")))):
            values = frame.select(r=result)["r"]
            assert values.null_count() == len(_OUT_OF_RANGE), f"{spec.name} {regime} {name}: {values.to_list()}"


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.name)
@given(data=st.data())
def test_ppf_and_isf_are_not_null_at_the_closed_endpoints(spec: DistSpec, data: st.DataObject) -> None:
    """`q` of exactly `0` or `1` is in range: the result is a support bound, never null."""
    params = data.draw(spec.params)
    frame = pl.DataFrame({"q": _ENDPOINTS}, schema={"q": pl.Float64})

    for regime, dist in _both_regimes(spec, params):
        for name, result in (("ppf", dist.ppf(pl.col("q"))), ("isf", dist.isf(pl.col("q")))):
            values = frame.select(r=result)["r"]
            assert values.null_count() == 0, f"{spec.name} {regime} {name}: {values.to_list()}"
