from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from hypothesis import strategies as st

from polars_stats import Bernoulli, Normal, Uniform

if TYPE_CHECKING:
    from collections.abc import Callable

    import polars as pl
    from hypothesis.strategies import SearchStrategy

    from polars_stats.distributions._base import _UnivariateDistribution


def _finite(min_value: float, max_value: float) -> SearchStrategy[float]:
    """Bounded, non-degenerate floats (no NaN/inf), the only inputs a parameter sweep should explore."""
    return st.floats(min_value=min_value, max_value=max_value, allow_nan=False, allow_infinity=False)


@dataclass(frozen=True)
class DistSpec:
    """Describes one distribution for the property suite: how to build it and where it is defined.

    A single `DistSpec` lets each property be written once and parametrised across every distribution.
    `params` draws a *valid* parameterisation (samplers and closed forms never raise on these), `make`
    turns that tuple into an instance, and `density` selects `pdf` (continuous) or `pmf` (discrete) so
    the non-negativity / normalisation properties read the same for both families.

    Arguments:
        name: Test id.
        continuous: `True` for a continuous distribution (has `pdf`, finite-grid integral), `False` for
            a discrete one (has `pmf`, finite-support sum).
        params: Strategy yielding a valid parameter tuple.
        make: Builds an instance from a parameter tuple.
        density: `pdf` or `pmf` as `(dist, expr) -> expr`. Typed loosely because the method differs by family;
            the concrete distribution is fixed per spec.
        eval_range: `(lo, hi)` finite window for evaluating cdf / density on a grid.
        integration_bounds: `(lo, hi)` over which the pdf integrates to ~1 (continuous only);
            the normal is truncated to a wide multiple of `std`. `None` for discrete.
        support: Finite list of mass points summing to ~1 (discrete only). `None` for continuous.
    """

    name: str
    continuous: bool
    params: SearchStrategy[tuple[float, ...]]
    make: Callable[[tuple[float, ...]], _UnivariateDistribution]
    density: Callable[[Any, pl.Expr], pl.Expr]
    eval_range: Callable[[tuple[float, ...]], tuple[float, float]]
    integration_bounds: Callable[[tuple[float, ...]], tuple[float, float]] | None = None
    support: Callable[[tuple[float, ...]], list[float]] | None = None


_BERNOULLI = DistSpec(
    name="bernoulli",
    continuous=False,
    params=st.tuples(_finite(0.0, 1.0)),
    make=lambda p: Bernoulli(p=p[0]),
    density=lambda d, c: d.pmf(c),
    eval_range=lambda _: (-1.0, 2.0),
    support=lambda _: [0.0, 1.0],
)

_NORMAL = DistSpec(
    name="normal",
    continuous=True,
    params=st.tuples(_finite(-10.0, 10.0), _finite(1e-2, 10.0)),
    make=lambda p: Normal(mean=p[0], std_dev=p[1]),
    density=lambda d, c: d.pdf(c),
    eval_range=lambda p: (p[0] - 6.0 * p[1], p[0] + 6.0 * p[1]),
    integration_bounds=lambda p: (p[0] - 12.0 * p[1], p[0] + 12.0 * p[1]),
)

_UNIFORM = DistSpec(
    name="uniform",
    continuous=True,
    # Draw `(min, width)` with `width > 0`, then expose `(min, max)`; this guarantees `max > min`
    # without rejection, so every drawn parameterisation is valid.
    params=st.tuples(_finite(-10.0, 10.0), _finite(1e-2, 20.0)).map(lambda mw: (mw[0], mw[0] + mw[1])),
    make=lambda p: Uniform(min=p[0], max=p[1]),
    density=lambda d, c: d.pdf(c),
    eval_range=lambda p: (p[0] - 0.5 * (p[1] - p[0]), p[1] + 0.5 * (p[1] - p[0])),
    integration_bounds=lambda p: (p[0], p[1]),
)

ALL_SPECS = [_BERNOULLI, _NORMAL, _UNIFORM]
CONTINUOUS_SPECS = [s for s in ALL_SPECS if s.continuous]
DISCRETE_SPECS = [s for s in ALL_SPECS if not s.continuous]
