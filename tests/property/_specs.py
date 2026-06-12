from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import TYPE_CHECKING, Any

import polars as pl
from hypothesis import strategies as st

from polars_stats import Bernoulli, Binomial, LogNormal, Normal, Uniform

if TYPE_CHECKING:
    from collections.abc import Callable

    from hypothesis.strategies import SearchStrategy

    from polars_stats._typing import PolarsDataType
    from polars_stats.distributions._base import _UnivariateDistribution


def _col(value: float, dtype: PolarsDataType | None = None) -> pl.Expr:
    """A full-length constant column expr (`pl.repeat`), forcing the per-row sampler path.

    A Python scalar would route through the constant-parameter fast path; wrapping it as an `Expr`
    instead drives the general per-row plugin, so a spec can build both variants from one parameter
    tuple and a test can assert the two paths agree.
    """
    return pl.repeat(value, n=pl.len(), dtype=dtype)


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
        make: Builds an instance from a parameter tuple (scalar parameters: the constant-parameter fast path).
        make_columns: Builds the same instance with each parameter wrapped as a full-length column expr,
            forcing the general per-row sampler path. Lets a test assert the fast path matches it draw-for-draw.
        make_masked: Like `make_columns`, but the first parameter is null on rows where the given boolean
            mask expr is true (null in any one parameter is enough to null a sampler row). Lets a test
            assert the per-row path's null contract without knowing the distribution's parameter names.
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
    make_columns: Callable[[tuple[float, ...]], _UnivariateDistribution]
    make_masked: Callable[[tuple[float, ...], pl.Expr], _UnivariateDistribution]
    density: Callable[[Any, pl.Expr], pl.Expr]
    eval_range: Callable[[tuple[float, ...]], tuple[float, float]]
    integration_bounds: Callable[[tuple[float, ...]], tuple[float, float]] | None = None
    support: Callable[[tuple[float, ...]], list[float]] | None = None


_BERNOULLI = DistSpec(
    name="bernoulli",
    continuous=False,
    params=st.tuples(_finite(0.0, 1.0)),
    make=lambda p: Bernoulli(p=p[0]),
    make_columns=lambda p: Bernoulli(p=_col(p[0])),
    make_masked=lambda p, m: Bernoulli(p=pl.when(~m).then(_col(p[0]))),
    density=lambda d, c: d.pmf(c),
    eval_range=lambda _: (-1.0, 2.0),
    support=lambda _: [0.0, 1.0],
)

_BINOMIAL = DistSpec(
    name="binomial",
    continuous=False,
    # `n` is drawn as an integer trial count; `p` spans the closed unit interval (incl. the
    # degenerate endpoints, where the mass collapses onto a single support point).
    params=st.tuples(st.integers(min_value=0, max_value=20), _finite(0.0, 1.0)),
    make=lambda p: Binomial(n=int(p[0]), p=p[1]),
    make_columns=lambda p: Binomial(n=_col(int(p[0]), pl.Int64()), p=_col(p[1])),
    make_masked=lambda p, m: Binomial(n=pl.when(~m).then(_col(int(p[0]), pl.Int64())), p=_col(p[1])),
    density=lambda d, c: d.pmf(c),
    eval_range=lambda p: (-1.0, p[0] + 1.0),
    support=lambda p: [float(k) for k in range(int(p[0]) + 1)],
)

_NORMAL = DistSpec(
    name="normal",
    continuous=True,
    params=st.tuples(_finite(-10.0, 10.0), _finite(1e-2, 10.0)),
    make=lambda p: Normal(mean=p[0], std_dev=p[1]),
    make_columns=lambda p: Normal(mean=_col(p[0]), std_dev=_col(p[1])),
    make_masked=lambda p, m: Normal(mean=pl.when(~m).then(_col(p[0])), std_dev=_col(p[1])),
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
    make_columns=lambda p: Uniform(min=_col(p[0]), max=_col(p[1])),
    make_masked=lambda p, m: Uniform(min=pl.when(~m).then(_col(p[0])), max=_col(p[1])),
    density=lambda d, c: d.pdf(c),
    eval_range=lambda p: (p[0] - 0.5 * (p[1] - p[0]), p[1] + 0.5 * (p[1] - p[0])),
    integration_bounds=lambda p: (p[0], p[1]),
)

_LOGNORMAL = DistSpec(
    name="lognormal",
    continuous=True,
    # `sigma` is capped at 0.9: the heavy right tail makes a uniform-grid trapezoidal integral lose
    # accuracy as `sigma` grows, and past ~1.0 the 4096-point mass check drifts above the 1e-3
    # tolerance. The functional and scipy-parity suites cover larger `sigma` directly.
    params=st.tuples(_finite(-1.5, 1.5), _finite(0.1, 0.9)),
    make=lambda p: LogNormal(mu=p[0], sigma=p[1]),
    make_columns=lambda p: LogNormal(mu=_col(p[0]), sigma=_col(p[1])),
    make_masked=lambda p, m: LogNormal(mu=pl.when(~m).then(_col(p[0])), sigma=_col(p[1])),
    density=lambda d, c: d.pdf(c),
    # Support is (0, inf); the grid stays on the positive side and out to a 4-sigma-in-log upper tail.
    eval_range=lambda p: (0.0, exp(p[0] + 4.0 * p[1])),
    integration_bounds=lambda p: (0.0, exp(p[0] + 6.0 * p[1])),
)

ALL_SPECS = [_BERNOULLI, _BINOMIAL, _NORMAL, _UNIFORM, _LOGNORMAL]
CONTINUOUS_SPECS = [s for s in ALL_SPECS if s.continuous]
DISCRETE_SPECS = [s for s in ALL_SPECS if not s.continuous]
