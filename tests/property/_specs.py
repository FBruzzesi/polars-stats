from __future__ import annotations

from dataclasses import dataclass
from math import ceil, exp, log, log1p
from typing import TYPE_CHECKING, Any

import polars as pl
from hypothesis import strategies as st

from polars_stats import Bernoulli, Beta, Binomial, Exponential, Geometric, LogNormal, Normal, Uniform

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


def _lit(value: float, dtype: PolarsDataType | None = None) -> pl.Expr:
    """The same constant as `_col` but length 1: the input `align_inputs` has to broadcast."""
    return pl.lit(value, dtype=dtype)


SERIES_ROWS = 64
"""Length of a `_series` parameter. Longer than any frame a test evaluates it on, and long enough that
a per-row seeded sampler will not repeat one value by chance."""


def _series(value: float, dtype: PolarsDataType | None = None) -> pl.Expr:
    """The same constant as `_col` but a fixed-length `pl.Series` literal, independent of frame height.

    `_col` follows `pl.len()` and can never outrun the frame. This one can, so the sampler's row
    index is the *shorter* input."""
    return pl.lit(pl.Series([value] * SERIES_ROWS, dtype=dtype))


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
        make_literals: Like `make_columns`, but every parameter is a length-1 `pl.lit`, so the plugin has to
            broadcast. Must agree with `make_columns` row for row.
        make_series: Like `make_columns`, but every parameter is a `SERIES_ROWS`-long `pl.Series` literal, so
            the parameters outrun the frame and set the call's row count.
        example: One valid parameterisation, for tests that need a fixed frame rather than a sweep.
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
    make_literals: Callable[[tuple[float, ...]], _UnivariateDistribution]
    make_series: Callable[[tuple[float, ...]], _UnivariateDistribution]
    example: tuple[float, ...]
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
    make_literals=lambda p: Bernoulli(p=_lit(p[0])),
    make_series=lambda p: Bernoulli(p=_series(p[0])),
    example=(0.3,),
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
    make_literals=lambda p: Binomial(n=_lit(int(p[0]), pl.Int64()), p=_lit(p[1])),
    make_series=lambda p: Binomial(n=_series(int(p[0]), pl.Int64()), p=_series(p[1])),
    example=(7, 0.35),
    density=lambda d, c: d.pmf(c),
    eval_range=lambda p: (-1.0, p[0] + 1.0),
    support=lambda p: [float(k) for k in range(int(p[0]) + 1)],
)


def _geometric_support_limit(p: float) -> int:
    """Smallest `k` whose tail mass `(1 - p)**k` falls below `1e-15`.

    The geometric support is infinite, so the suite truncates it where the missing tail sits far
    below its mass tolerance. At the degenerate `p = 1` the whole mass sits on `k = 1`."""
    return 1 if p >= 1.0 else ceil(log(1e-15) / log1p(-p))


_GEOMETRIC = DistSpec(
    name="geometric",
    continuous=False,
    # `p` spans `(0, 1]`: unlike Bernoulli there is no `p = 0` point mass, since "never succeeds"
    # has no moments and statrs rejects it.
    params=st.tuples(_finite(0.05, 1.0)),
    make=lambda p: Geometric(p=p[0]),
    make_columns=lambda p: Geometric(p=_col(p[0])),
    make_masked=lambda p, m: Geometric(p=pl.when(~m).then(_col(p[0]))),
    make_literals=lambda p: Geometric(p=_lit(p[0])),
    make_series=lambda p: Geometric(p=_series(p[0])),
    example=(0.3,),
    density=lambda d, c: d.pmf(c),
    eval_range=lambda p: (-1.0, float(_geometric_support_limit(p[0]) + 1)),
    support=lambda p: [float(k) for k in range(1, _geometric_support_limit(p[0]) + 1)],
)

_NORMAL = DistSpec(
    name="normal",
    continuous=True,
    params=st.tuples(_finite(-10.0, 10.0), _finite(1e-2, 10.0)),
    make=lambda p: Normal(mu=p[0], sigma=p[1]),
    make_columns=lambda p: Normal(mu=_col(p[0]), sigma=_col(p[1])),
    make_masked=lambda p, m: Normal(mu=pl.when(~m).then(_col(p[0])), sigma=_col(p[1])),
    make_literals=lambda p: Normal(mu=_lit(p[0]), sigma=_lit(p[1])),
    make_series=lambda p: Normal(mu=_series(p[0]), sigma=_series(p[1])),
    example=(1.5, 2.0),
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
    make_literals=lambda p: Uniform(min=_lit(p[0]), max=_lit(p[1])),
    make_series=lambda p: Uniform(min=_series(p[0]), max=_series(p[1])),
    example=(-1.0, 3.0),
    density=lambda d, c: d.pdf(c),
    eval_range=lambda p: (p[0] - 0.5 * (p[1] - p[0]), p[1] + 0.5 * (p[1] - p[0])),
    integration_bounds=lambda p: (p[0], p[1]),
)

_LOGNORMAL = DistSpec(
    name="lognormal",
    continuous=True,
    # `sigma` is capped at 0.9: the heavy right tail makes a uniform-grid trapezoidal integral lose
    # accuracy as `sigma` grows, and past ~1.2 the mass check drifts above the 1e-3 tolerance at the
    # current `_INTEGRATION_GRID_SIZE`. The functional and scipy-parity suites cover larger `sigma`.
    params=st.tuples(_finite(-1.5, 1.5), _finite(0.1, 0.9)),
    make=lambda p: LogNormal(mu=p[0], sigma=p[1]),
    make_columns=lambda p: LogNormal(mu=_col(p[0]), sigma=_col(p[1])),
    make_masked=lambda p, m: LogNormal(mu=pl.when(~m).then(_col(p[0])), sigma=_col(p[1])),
    make_literals=lambda p: LogNormal(mu=_lit(p[0]), sigma=_lit(p[1])),
    make_series=lambda p: LogNormal(mu=_series(p[0]), sigma=_series(p[1])),
    example=(0.5, 0.75),
    density=lambda d, c: d.pdf(c),
    # Support is (0, inf); the grid stays on the positive side and out to a 4-sigma-in-log upper tail.
    eval_range=lambda p: (0.0, exp(p[0] + 4.0 * p[1])),
    integration_bounds=lambda p: (0.0, exp(p[0] + 6.0 * p[1])),
)

_EXPONENTIAL = DistSpec(
    name="exponential",
    continuous=True,
    # `rate > 0`; the lower bound keeps the mean `1 / rate` finite enough for the trapezoidal mass
    # check, the upper bound keeps it from collapsing onto a near-degenerate spike at 0.
    params=st.tuples(_finite(1e-2, 10.0)),
    make=lambda p: Exponential(rate=p[0]),
    make_columns=lambda p: Exponential(rate=_col(p[0])),
    make_masked=lambda p, m: Exponential(rate=pl.when(~m).then(_col(p[0]))),
    make_literals=lambda p: Exponential(rate=_lit(p[0])),
    make_series=lambda p: Exponential(rate=_series(p[0])),
    example=(1.5,),
    density=lambda d, c: d.pdf(c),
    # Support is [0, inf); the grid spans the `x < 0` zero region through several means.
    eval_range=lambda p: (-1.0 / p[0], 5.0 / p[0]),
    # `1 - exp(-30)` of the mass lies below `30 / rate`; the rest is below the 1e-3 tolerance.
    integration_bounds=lambda p: (0.0, 30.0 / p[0]),
)

_BETA = DistSpec(
    name="beta",
    continuous=True,
    # Shapes are kept >= 1 so the density is finite at the support endpoints: below 1 it diverges
    # there, making `pdf(0)` or `pdf(1)` inf and the trapezoidal mass integral inf with it. Finite
    # is not the same as accurately integrable: just above 1 the density stays high right up to the
    # endpoint and the rule drops the final half-cell, which is what sizes `_INTEGRATION_GRID_SIZE`.
    # Shapes below 1 are covered directly by the functional and scipy-parity suites.
    params=st.tuples(_finite(1.0, 10.0), _finite(1.0, 10.0)),
    make=lambda p: Beta(a=p[0], b=p[1]),
    make_columns=lambda p: Beta(a=_col(p[0]), b=_col(p[1])),
    make_masked=lambda p, m: Beta(a=pl.when(~m).then(_col(p[0])), b=_col(p[1])),
    make_literals=lambda p: Beta(a=_lit(p[0]), b=_lit(p[1])),
    make_series=lambda p: Beta(a=_series(p[0]), b=_series(p[1])),
    example=(2.0, 3.0),
    density=lambda d, c: d.pdf(c),
    # Support is [0, 1]; the grid spans the zero-density regions on both sides.
    eval_range=lambda _: (-0.5, 1.5),
    integration_bounds=lambda _: (0.0, 1.0),
)

ALL_SPECS = [_BERNOULLI, _BINOMIAL, _GEOMETRIC, _NORMAL, _UNIFORM, _LOGNORMAL, _EXPONENTIAL, _BETA]
CONTINUOUS_SPECS = [s for s in ALL_SPECS if s.continuous]
DISCRETE_SPECS = [s for s in ALL_SPECS if not s.continuous]

# These specs compute in Polars, not in a Rust body, so with constant parameters their operands are
# length-1 literals that polars may fold differently from the column kernel. Every IEEE operation is
# exactly rounded, so the measured 1-ULP gap (`Exponential.ppf`, Uniform's `variance` / `std`) is a
# different operation *order*, not a different formula. Extend from a failing assertion, never by
# widening the tolerance.
ULP_TOLERANT_VALUE_SPECS = frozenset({"uniform", "exponential"})
"""Specs whose value-keyed methods compare to `ULP_REL_TOL` instead of bit-exactly."""

ULP_TOLERANT_MOMENT_SPECS = frozenset({"uniform", "geometric"})
"""Specs whose moments compare to `ULP_REL_TOL` instead of bit-exactly.

`geometric` is here for `std` and `entropy` only, and for a narrower reason than `uniform`: both
divide by the parameter itself, and `_col`'s `pl.repeat` keeps `p` in polars' scalar-backed
representation, whose division kernel is a reciprocal multiply rather than an exactly-rounded divide.
A materialised `p` column and a `pl.lit(pl.Series(...))` one both stay bit-exact against the fast
path (0 divergences over 300 random `p`); only the `pl.repeat` spelling moves the last bit.
"""

ULP_REL_TOL = 1e-15
"""~4x a double's ULP."""

ULP_ABS_TOL = 0.0
"""No absolute slack, so a tiny unequal pair cannot pass for free."""
