"""`ppf` and `isf` at the closed endpoints of the unit interval, where no parity grid reaches.

Every parity `_QUANTILES` list stops short of `0` and `1` (or, for `Uniform`, includes them without an
oracle for the discrete families), so `ppf(0)` and `ppf(1)` have no scipy reference. They have a better
one: the support bounds the registry already records. `isf` is the same claim with the ends swapped.

The domain contract around those endpoints (outside `[0, 1]` is null, `NaN` propagates) is
`tests/property/ppf_domain_test.py` and `validation_test.py`; the interior values are parity's.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from tests._registry import DISCRETE_SPECS, DRIVER_REGIMES

if TYPE_CHECKING:
    from tests._registry import DistSpec, Regime


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
@pytest.mark.parametrize("method", ["ppf", "isf"])
def test_the_closed_endpoints_are_the_support_bounds(spec: DistSpec, method: str, regime: Regime) -> None:
    """`ppf(0)` is the lower bound and `ppf(1)` the upper; `isf` answers them the other way round.

    Both directions diverge from scipy for the discrete families, whose generic `ppf(0)` reports the
    below-support sentinel `min - 1` where this clamps to the support. `tests/scipy_parity` therefore
    cannot state it, and its own `_QUANTILES` stay strictly inside.
    """
    lo, hi = spec.bounds
    quantiles = [0.0, 1.0]
    expected = [lo, hi] if method == "ppf" else [hi, lo]

    frame = pl.DataFrame({"q": quantiles}, schema={"q": pl.Float64})
    got = frame.select(r=getattr(spec.build(regime), method)(pl.col("q")))["r"].to_list()

    assert got == expected, f"{spec.name}.{method} at {quantiles}: {got} != {expected}"


@pytest.mark.parametrize("spec", DISCRETE_SPECS, ids=lambda spec: spec.name)
@pytest.mark.parametrize("regime", DRIVER_REGIMES)
@pytest.mark.parametrize("method", ["ppf", "isf"])
def test_a_discrete_inverse_is_integer_valued(spec: DistSpec, method: str, regime: Regime) -> None:
    """A discrete inverse lands on a support point, so it has no fractional part.

    Parity compares the *value* against scipy at a tolerance, which a `0.5`-off answer inside the
    tolerance band would pass; this is the exactness claim that comparison cannot make.

    Parametrised over `DISCRETE_SPECS` rather than skipping the continuous rows, so the node count
    is the number of claims actually made.
    """
    quantiles = [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]
    frame = pl.DataFrame({"q": quantiles}, schema={"q": pl.Float64})
    got = frame.select(r=getattr(spec.build(regime), method)(pl.col("q")))["r"]

    assert (got == got.floor()).all(), f"{spec.name}.{method} answered off the support: {got.to_list()}"
