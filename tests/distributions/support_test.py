"""Outside the support, every value-keyed answer is a constant, and the six of them agree.

One contract spanning six methods, which is why it is one file rather than a paragraph repeated in
`cdf_test.py`, `sf_test.py` and their logs: below the support the mass is `0` and the survival `1`,
above it the reverse, and the density is `0` on both sides. Splitting that by method would scatter a
single claim across four files and let three of them drift.

The points come from `spec.bounds`, so a row with a wrong bound fails here rather than passing
vacuously; that is the behavioural half of what `registry_test.py` cannot check structurally.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import polars as pl
import pytest

from tests._registry import CONTINUOUS_SPECS, DRIVER_REGIMES, density, log_density

if TYPE_CHECKING:
    from collections.abc import Callable

    from polars_stats.distributions._base import _UnivariateDistribution
    from tests._registry import DistSpec, Regime

_NEG_INF = -math.inf

# Method -> (value below the support, value above it). `log_sf` is `log(1) = 0` below and `-inf`
# above; `log_cdf` the mirror. The density is `0` on both sides, so its log is `-inf` on both.
_EXPECTED: dict[str, tuple[float, float]] = {
    "cdf": (0.0, 1.0),
    "log_cdf": (_NEG_INF, 0.0),
    "sf": (1.0, 0.0),
    "log_sf": (0.0, _NEG_INF),
    "density": (0.0, 0.0),
    "log_density": (_NEG_INF, _NEG_INF),
}


def _call(dist: _UnivariateDistribution, method: str, value: pl.Expr) -> pl.Expr:
    """One method by name, the family-specific density pair reached through the registry helpers."""
    if method == "density":
        return density(dist, value)
    if method == "log_density":
        return log_density(dist, value)
    call: Callable[[pl.Expr], pl.Expr] = getattr(dist, method)
    return call(value)


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
@pytest.mark.parametrize("method", list(_EXPECTED))
def test_off_support_answers_are_the_saturated_constants(spec: DistSpec, method: str, regime: Regime) -> None:
    """Below the support and above it, on whichever sides the support is bounded.

    A distribution unbounded on both sides has no off-support point to probe and says so, rather than
    passing with an empty loop.
    """
    if not spec.off_support_points:
        pytest.skip(f"{spec.name} has no off-support point: its support is the whole line")

    dist = spec.build(regime)
    below, above = _EXPECTED[method]
    points = {x: (below if x < spec.bounds[0] else above) for x in spec.off_support_points}

    frame = pl.DataFrame({"x": list(points)}, schema={"x": pl.Float64})
    got = frame.select(r=_call(dist, method, pl.col("x")))["r"].to_list()

    assert got == list(points.values()), f"{spec.name}.{method} off-support: {got} != {list(points.values())}"


# The four cumulative methods at a *finite support endpoint* of a continuous distribution, where no
# mass sits: the cdf has not started at the lower bound and has finished at the upper one. The density
# is absent on purpose, since its value at an endpoint is the distribution's own (`0` for `Beta(2, 3)`
# and `LogNormal`, `1 / width` for `Uniform`, `rate` for `Exponential`).
#
# Discrete rows are excluded because `cdf(lo)` is `pmf(lo)`, not `0`.
_AT_ENDPOINT: dict[str, tuple[float, float]] = {
    "cdf": (0.0, 1.0),
    "log_cdf": (_NEG_INF, 0.0),
    "sf": (1.0, 0.0),
    "log_sf": (0.0, _NEG_INF),
}


@pytest.mark.parametrize("spec", CONTINUOUS_SPECS, ids=lambda spec: spec.name)
@pytest.mark.parametrize("regime", DRIVER_REGIMES)
@pytest.mark.parametrize("method", list(_AT_ENDPOINT))
def test_a_finite_endpoint_of_a_continuous_support_is_already_saturated(
    spec: DistSpec, method: str, regime: Regime
) -> None:
    """`cdf(lo) == 0` and `cdf(hi) == 1` exactly, on whichever endpoints are finite.

    Not reachable from the parity grids: `Beta`'s stops at `0.01` / `0.99`, `Exponential`'s and
    `LogNormal`'s stay strictly inside the open support, so the endpoint itself has no scipy oracle.
    It is also where a half-open support shows itself: `LogNormal(0)` and `Exponential(0)` answer as
    the off-support side, matching `scipy.stats.lognorm` and `expon`.
    """
    dist = spec.build(regime)
    below, above = _AT_ENDPOINT[method]
    points = {x: want for x, want in zip(spec.bounds, (below, above), strict=True) if not math.isinf(x)}
    if not points:
        pytest.skip(f"{spec.name} has no finite support endpoint")

    frame = pl.DataFrame({"x": list(points)}, schema={"x": pl.Float64})
    got = frame.select(r=getattr(dist, method)(pl.col("x")))["r"].to_list()

    assert got == list(points.values()), f"{spec.name}.{method} at its endpoints: {got} != {list(points.values())}"


# The density *at* a finite endpoint, which the shared rule above cannot state: its value is the
# distribution's own, not a saturated constant. Written out per distribution because that is what it
# is, and it is what catches a `<` where the support branch means `<=`. `LogNormal` and `Exponential`
# are the two half-open supports, and they disagree at the same point: the lognormal excludes `0`
# (matching `scipy.stats.lognorm`), the exponential includes it.
#
# name -> {endpoint: (density, log_density)}
_DENSITY_AT_ENDPOINT: dict[str, dict[float, tuple[float, float]]] = {
    "uniform": {-1.0: (0.25, math.log(0.25)), 3.0: (0.25, math.log(0.25))},
    "lognormal": {0.0: (0.0, _NEG_INF)},
    "exponential": {0.0: (1.5, math.log(1.5))},
    "pareto": {1.5: (2.0, math.log(2.0))},
    "beta": {0.0: (0.0, _NEG_INF), 1.0: (0.0, _NEG_INF)},
    "bernoulli": {0.0: (0.7, math.log(0.7)), 1.0: (0.3, math.log(0.3))},
    "binomial": {0.0: (0.65**7, 7 * math.log(0.65)), 7.0: (0.35**7, 7 * math.log(0.35))},
    "discreteuniform": {1.0: (1 / 6, math.log(1 / 6)), 6.0: (1 / 6, math.log(1 / 6))},
    "geometric": {1.0: (0.3, math.log(0.3))},
}


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
def test_the_density_at_a_finite_endpoint_is_the_recorded_value(spec: DistSpec, regime: Regime) -> None:
    """Every finite endpoint of `spec.bounds` at `example` has a recorded density, and it is right."""
    finite = [x for x in spec.bounds if not math.isinf(x)]
    if not finite:
        pytest.skip(f"{spec.name} has no finite support endpoint")

    assert spec.name in _DENSITY_AT_ENDPOINT, f"{spec.name} has finite endpoints {finite} but no recorded density"
    recorded = _DENSITY_AT_ENDPOINT[spec.name]
    assert sorted(recorded) == sorted(finite), f"{spec.name}: recorded {sorted(recorded)} for endpoints {finite}"

    dist = spec.build(regime)
    frame = pl.DataFrame({"x": list(recorded)}, schema={"x": pl.Float64})
    got = frame.select(d=density(dist, pl.col("x")), ld=log_density(dist, pl.col("x")))

    assert got["d"].to_list() == pytest.approx([v[0] for v in recorded.values()], rel=1e-14)
    assert got["ld"].to_list() == pytest.approx([v[1] for v in recorded.values()], rel=1e-14)
