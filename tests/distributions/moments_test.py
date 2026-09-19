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
from tests._registry import (
    DRIVER_REGIMES,
    MOMENTS,
    SPECS_BY_NAME,
    ULP_ABS_TOL,
    ULP_REL_TOL,
    UNDEFINED_MOMENTS,
    compares_bit_exactly,
    moment_is_undefined,
)

if TYPE_CHECKING:
    from polars_stats._typing import DistributionName
    from tests._registry import DistSpec, Moment, Regime

_UNDEFINED_PAIRS = [
    (spec_name, moment) for spec_name, moments in UNDEFINED_MOMENTS.items() for moment in sorted(moments)
]
"""`UNDEFINED_MOMENTS` flattened, so the dtype contract below has one node per pair rather than a skip per spec."""


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

    A mean recorded in `UNDEFINED_MOMENTS` is pinned to **null** instead, on both routings: `Cauchy` has
    no mean, and `null` (not `NaN`, not a raise) is the contract for that.
    """
    lo, hi = spec.bounds
    got = pl.DataFrame({"_": range(4)}).select(m=spec.build(regime).mean())["m"][0]
    if moment_is_undefined(spec.name, "mean"):
        assert got is None, f"{spec.name}: mean is recorded undefined but answered {got}"
        return
    assert math.isfinite(got), f"{spec.name}: mean is {got}"
    assert lo <= got <= hi, f"{spec.name}: mean {got} is outside {spec.bounds}"


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
def test_the_variance_is_non_negative(spec: DistSpec, regime: Regime) -> None:
    """A negative variance is a sign error; `std` would then be `NaN` rather than obviously wrong.

    A variance recorded in `UNDEFINED_MOMENTS` is pinned to **null** instead, as the mean above.
    """
    got = pl.DataFrame({"_": range(4)}).select(v=spec.build(regime).variance())["v"][0]
    if moment_is_undefined(spec.name, "variance"):
        assert got is None, f"{spec.name}: variance is recorded undefined but answered {got}"
        return
    assert got >= 0.0, f"{spec.name}: variance {got} is negative"


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
@pytest.mark.parametrize(("spec_name", "moment"), _UNDEFINED_PAIRS, ids=str)
def test_an_undefined_moment_is_a_typed_null_not_a_null_column(
    spec_name: DistributionName, moment: Moment, regime: Regime
) -> None:
    """An undefined moment is a `Float64` null, not a `Null`-typed column.

    A `Null` column widens the schema of whatever it is concatenated, joined or written into, where a
    typed null does not. No value assertion sees the difference, so nothing else pins the dtype that
    `_cauchy.py`'s `_UNDEFINED_MOMENT` sets.
    """
    expr = getattr(SPECS_BY_NAME[spec_name].build(regime), moment)()
    got = pl.DataFrame({"_": range(4)}).select(m=expr)["m"]
    assert got.dtype == pl.Float64, f"{spec_name}.{moment}() is {got.dtype}, want Float64"
    assert got[0] is None


# The other half of `UNDEFINED_MOMENTS`: a divergent moment is `+inf`. Both thresholds are closed,
# as in `scipy.stats.pareto`: `shape = 1` has no mean, `shape = 2` no variance.
_DIVERGENT_MOMENTS_BY_SHAPE = [
    pytest.param(0.5, ("mean", "variance", "std"), id="shape=0.5"),
    pytest.param(1.0, ("mean", "variance", "std"), id="shape=1"),
    pytest.param(1.5, ("variance", "std"), id="shape=1.5"),
    pytest.param(2.0, ("variance", "std"), id="shape=2"),
    pytest.param(2.5, (), id="shape=2.5"),
]


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
@pytest.mark.parametrize(("shape", "divergent"), _DIVERGENT_MOMENTS_BY_SHAPE)
def test_a_divergent_pareto_moment_is_positive_infinity(
    regime: Regime, shape: float, divergent: tuple[Moment, ...]
) -> None:
    """`+inf` where the integral diverges, finite elsewhere, on both routings.

    Null is reserved for a moment with no value at all (`Cauchy`) and a raise for an invalid
    parameterisation, which `shape = 0.5` is not; `median` and `entropy` stay finite on every row.
    """
    dist = SPECS_BY_NAME["pareto"].build(regime, params=(1.5, shape))
    got = pl.DataFrame({"_": range(4)}).select(**{moment: getattr(dist, moment)() for moment in MOMENTS})
    for moment in MOMENTS:
        value = got[moment][0]
        if moment in divergent:
            assert value == math.inf, f"{moment} at shape={shape}: {value}"
        else:
            assert math.isfinite(value), f"{moment} at shape={shape}: {value}"
