"""The contracts that hold between the moments, which comparing each one to scipy cannot state.

`tests/scipy_parity` pins every moment's *value* at a tolerance and
`tests/property/moment_test.py` pins the constant-parameter path against the per-row one bit for bit.
Neither says anything about how two moments relate, and a `std` and a `variance` can both sit inside
their tolerances while disagreeing with each other.

The null and invalid-parameter behaviour of the moments is `validation_test.py`'s;
`fast_path_test.py` owns the routing.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import polars as pl
import pytest

from tests._polars_compat import assert_series_equal
from tests._registry import DRIVER_REGIMES, ULP_ABS_TOL, ULP_REL_TOL, compares_bit_exactly

if TYPE_CHECKING:
    from tests._registry import DistSpec, Regime


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
def test_std_is_the_square_root_of_variance(spec: DistSpec, regime: Regime) -> None:
    """`std()` is `variance().sqrt()`, which the base class composes and a subclass may override.

    `Geometric` does override it, precisely because the composition saturates where the direct form
    does not, so this is the assertion that its override still agrees in the ordinary range.
    """
    frame = pl.DataFrame({"_": range(4)})
    dist = spec.build(regime)

    got = frame.select(std=dist.std(), root=dist.variance().sqrt())
    exact = compares_bit_exactly(spec.name, "std")
    assert_series_equal(
        got["std"], got["root"], check_names=False, check_exact=exact, rel_tol=ULP_REL_TOL, abs_tol=ULP_ABS_TOL
    )


def test_the_median_agrees_across_parameter_routings(spec: DistSpec) -> None:
    """The one moment with no other link between the two routings.

    `tests/scipy_parity` pins `median` on the scalar routing only, and `property/moment_test.py`
    excludes it because the discrete families solve it through `ppf`. That left the continuous closed
    forms (`(min + max) / 2`, `log(2) / rate`, `exp(mu)`) asserted on one routing and not the other,
    which is what the deleted per-distribution column tests used to cover.

    Not parametrised over `DRIVER_REGIMES`: the comparison *is* the two regimes, so sweeping them
    would compare the scalar routing against itself for half the nodes.
    """
    frame = pl.DataFrame({"_": range(4)})
    got = frame.select(scalar=spec.build("scalar").median(), column=spec.build("column").median())
    exact = compares_bit_exactly(spec.name, "median")
    assert_series_equal(
        got["scalar"], got["column"], check_names=False, check_exact=exact, rel_tol=ULP_REL_TOL, abs_tol=ULP_ABS_TOL
    )


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
def test_the_mean_lies_inside_the_support(spec: DistSpec, regime: Regime) -> None:
    """A mean outside the bounds is a swapped or misread parameter, which no tolerance would catch.

    Finiteness is asserted alongside, since `Normal`'s bounds are the whole line and the containment
    claim alone would be vacuous for it.
    """
    lo, hi = spec.bounds
    got = pl.DataFrame({"_": range(4)}).select(m=spec.build(regime).mean())["m"][0]
    assert math.isfinite(got), f"{spec.name}: mean is {got}"
    assert lo <= got <= hi, f"{spec.name}: mean {got} is outside {spec.bounds}"


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
def test_the_variance_is_non_negative(spec: DistSpec, regime: Regime) -> None:
    """A negative variance is a sign error; `std` would then be `NaN` rather than obviously wrong."""
    got = pl.DataFrame({"_": range(4)}).select(v=spec.build(regime).variance())["v"][0]
    assert got >= 0.0, f"{spec.name}: variance {got} is negative"
