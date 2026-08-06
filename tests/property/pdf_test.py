from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests._polars_compat import linear_space
from tests.property._specs import ALL_SPECS, CONTINUOUS_SPECS, DISCRETE_SPECS

if TYPE_CHECKING:
    from collections.abc import Iterable

    from tests.property._specs import DistSpec

_GRID_SIZE = 64
# Grid and tolerance for the trapezoidal mass check. Normal, Uniform and Exponential integrate to
# ~1e-6 or better here; the binding case is a density still well above zero at a support endpoint,
# where the rule drops the final half-cell. Beta with one shape at 1 and the other at `s` is the
# worst of those, with error almost exactly `s / (2 * (grid - 1))`. Shapes are capped at 10 in
# `_specs.py`, so grid 4096 gives 1.2e-3 and fails the tolerance; 16384 gives 3.1e-4, a ~3x margin.
_INTEGRATION_GRID_SIZE = 16384
_INTEGRATION_TOL = 1e-3


def _eval(expr: pl.Expr, xs: Iterable[float]) -> pl.Series:
    return pl.DataFrame({"x": xs}).select(r=expr)["r"]


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.name)
@given(data=st.data())
def test_density_non_negative(spec: DistSpec, data: st.DataObject) -> None:
    """`pdf(x) >= 0` (continuous) / `pmf(x) >= 0` (discrete), across the parameter space.

    A trailing `NaN` evaluation point must propagate as `NaN` (scipy semantics) rather than collapse
    into the zero-density branch; the finite-grid assertions exclude it.
    """
    params = data.draw(spec.params)
    dist = spec.make(params)
    lo, hi = spec.eval_range(params)
    xs = linear_space(lo, hi, _GRID_SIZE)

    density = _eval(spec.density(dist, pl.col("x")), [*xs, float("nan")])

    assert math.isnan(density.item(_GRID_SIZE))
    density = density.head(_GRID_SIZE)
    assert not density.is_nan().any()
    assert density.ge(0.0).all()


@pytest.mark.parametrize("spec", CONTINUOUS_SPECS, ids=lambda s: s.name)
@settings(max_examples=25)
@given(data=st.data())
def test_pdf_integrates_to_one(spec: DistSpec, data: st.DataObject) -> None:
    """Trapezoidal integral of the pdf over the (truncated) support is ~1."""
    assert spec.integration_bounds is not None  # invariant for continuous specs
    params = data.draw(spec.params)
    dist = spec.make(params)
    lo, hi = spec.integration_bounds(params)
    xs = linear_space(lo, hi, _INTEGRATION_GRID_SIZE)

    density = _eval(spec.density(dist, pl.col("x")), xs)
    mass = float(np.trapezoid(density.to_numpy(), xs.to_numpy()))

    assert mass == pytest.approx(1.0, abs=_INTEGRATION_TOL)


@pytest.mark.parametrize("spec", DISCRETE_SPECS, ids=lambda s: s.name)
@given(data=st.data())
def test_pmf_sums_to_one(spec: DistSpec, data: st.DataObject) -> None:
    """Sum of the pmf over the finite support is ~1."""
    assert spec.support is not None  # invariant for discrete specs
    params = data.draw(spec.params)
    dist = spec.make(params)
    support = spec.support(params)

    mass = _eval(spec.density(dist, pl.col("x")), support).sum()

    assert mass == pytest.approx(1.0, abs=1e-12)
