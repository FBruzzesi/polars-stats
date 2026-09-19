"""One row per distribution: how to build it in every parameter regime, and where it is defined.

The suite's single registry. Every distribution-agnostic contract is written once and parametrised
over these rows, delivered by the fixtures in `tests/conftest.py`.
`tests/distributions/registry_test.py` is the guard that turns a missing row into a failing test.

What a row records and what it derives is in `tests/README.md`. Anything more than one module reads
lives here; a table only its own test file reads stays in that file.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, exp, inf, log, log1p, nextafter
from typing import TYPE_CHECKING, Literal, get_args

import polars as pl
from hypothesis import strategies as st

from polars_stats import (
    Bernoulli,
    Beta,
    Binomial,
    Cauchy,
    DiscreteUniform,
    Exponential,
    Geometric,
    LogNormal,
    Normal,
    Pareto,
    Uniform,
    is_continuous,
    is_discrete,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from hypothesis.strategies import SearchStrategy

    from polars_stats._typing import DistributionName, IntoExprColumn, PolarsDataType
    from polars_stats.distributions._base import _UnivariateDistribution

Regime = Literal["scalar", "column", "masked", "literal", "series", "name"]
"""The parameter spellings a distribution must agree across.

| Regime | The bug it catches |
| --- | --- |
| `scalar` | the constant-parameter fast path diverging from the general one |
| `column` | the general per-row plugin path |
| `masked` | a null in any one parameter must null that row |
| `literal` | neither engine broadcasts a length-1 plugin input on its own |
| `series` | the sampler's row index taken from the shorter input |
| `name` | a column-name `str` means `pl.col(name)`, never `pl.lit(name)` |
"""

SERIES_ROWS = 64
"""Length of a `series` parameter. Longer than any frame a test evaluates it on, and long enough that
a per-row seeded sampler will not repeat one value by chance."""

Method = Literal["pdf", "pmf", "log_pdf", "log_pmf", "cdf", "log_cdf", "sf", "log_sf", "ppf", "isf"]
"""Every value-keyed method name, so a mistyped one is a type error rather than an `AttributeError`."""

VALUE_KEYED_METHODS: tuple[Method, ...] = ("cdf", "log_cdf", "sf", "log_sf", "ppf", "isf")
"""The value-keyed methods every family has. The density pair is family-specific; see `density`."""


def point_column(method: Method) -> str:
    """The contract-frame column `method` reads.

    `ppf` and `isf` take a quantile in `[0, 1]`; every other value-keyed method takes a support point.
    """
    return "q" if method in ("ppf", "isf") else "x"


Moment = Literal["mean", "variance", "std", "median", "entropy"]
MOMENTS: tuple[Moment, ...] = get_args(Moment)

DRIVER_REGIMES: tuple[Regime, Regime] = ("scalar", "column")
"""The axis every shared contract sweeps: the two Rust driver families.

Constant parameters validate once and route through the `<name>_<method>_scalar` plugins; column
parameters validate per row and route through the general ones. The other four spellings in `Regime`
are broadcast and naming concerns, and belong to `broadcast_test.py` and `naming_test.py`."""


def density(dist: _UnivariateDistribution, value: pl.Expr | float) -> pl.Expr:
    """`pdf` (continuous) or `pmf` (discrete).

    Both guards are asked rather than one and an `else`: the density methods live on the family
    subclasses, so narrowing away `DiscreteDistribution` does not leave a type that has `pdf`.
    """
    if is_discrete(dist):
        return dist.pmf(value)
    if is_continuous(dist):
        return dist.pdf(value)
    msg = f"unsupported distribution family: {type(dist)}"  # pragma: no cover
    raise TypeError(msg)  # pragma: no cover


def log_density(dist: _UnivariateDistribution, value: pl.Expr | float) -> pl.Expr:
    """`log_pdf` (continuous) or `log_pmf` (discrete), narrowed the same way as `density`."""
    if is_discrete(dist):
        return dist.log_pmf(value)
    if is_continuous(dist):
        return dist.log_pdf(value)
    msg = f"unsupported distribution family: {type(dist)}"  # pragma: no cover
    raise TypeError(msg)  # pragma: no cover


def all_value_keyed(dist: _UnivariateDistribution) -> tuple[Method, ...]:
    """Every value-keyed method of `dist`, the family's density pair included."""
    density_pair: tuple[Method, Method] = ("pmf", "log_pmf") if is_discrete(dist) else ("pdf", "log_pdf")
    return (*density_pair, *VALUE_KEYED_METHODS)


def constant_column(value: float, dtype: PolarsDataType | None = None) -> pl.Expr:
    """A constant as a full-length column (`pl.repeat`), which forces the general per-row path.

    The same value as a Python scalar routes through the constant-parameter fast path instead, so one
    parameter tuple builds both and a test can assert the two agree.
    """
    return pl.repeat(value, n=pl.len(), dtype=dtype)


def min_max(drawn: pl.Series) -> tuple[float, float]:
    """`(min, max)` as plain floats.

    `Series.min()` is typed as a union spanning every dtype polars can hold, which no numeric
    comparison accepts; narrowing once here keeps the assertion sites readable.
    """
    low, high = drawn.cast(pl.Float64).min(), drawn.cast(pl.Float64).max()
    assert isinstance(low, float)
    assert isinstance(high, float)
    return low, high


def _finite(min_value: float, max_value: float) -> SearchStrategy[float]:
    """Bounded, non-degenerate floats (no NaN/inf), the only inputs a parameter sweep should explore."""
    return st.floats(min_value=min_value, max_value=max_value, allow_nan=False, allow_infinity=False)


@dataclass(frozen=True)
class Param:
    """One constructor parameter: its keyword, its column in the shared contract frame, and its type.

    Arguments:
        name: The constructor keyword (`mu`, `sigma`, `min`, `n`, ...). Also the parameter the
            validator names when it refuses a value, as `"<name> must be ..."`, which is what lets the
            non-finite sweep be derived rather than tabulated.
        column: Column in `CONTRACT_FRAME` below carrying a valid value for this
            parameter. Distinct per parameter within a distribution, which is what makes the
            output-naming assertions discriminating: an expression that followed `inputs[1]` instead of
            polars' first-input rule would name the output after the wrong column and fail.
        integer: `True` for a parameter spelled as an `int` and carried in an `Int64` column
            (`Binomial.n` via `coerce_n`, `DiscreteUniform`'s bounds via `coerce_int`). It accepts an
            `int` and rejects a `float`, the mirror image of every other parameter, so the constructor
            type assertions have to know which side a parameter is on.
        paired: `True` when this parameter's *domain* rule reads a sibling rather than the value alone
            (both `Uniform` bounds and both `DiscreteUniform` bounds, whose rule is an ordering). A
            paired rule cannot be probed beside a null sibling: with the sibling null there is no pair
            to compare, so the row nulls instead of raising, and that is the contract, not a defect.
            Every other parameter has a rule of its own, which survives any sibling being null.
    """

    name: str
    column: str
    integer: bool = False
    paired: bool = False

    @property
    def dtype(self) -> PolarsDataType:
        """The dtype every non-scalar spelling of this parameter carries."""
        return pl.Int64() if self.integer else pl.Float64()

    def cast(self, value: float) -> float:
        """`value` in this parameter's Python type: an `int` for an integer parameter, else unchanged.

        Not `float(value)` on the other branch: the coercers reject an `int` where a `float` is meant, so
        a row that spelled a float parameter as `0` should fail loudly in
        `registry_test.test_spec_builds_every_parameter_regime` rather than be repaired here.
        """
        return int(value) if self.integer else value


@dataclass(frozen=True)
class InvalidCase:
    """A *finite* parameterisation the plugins must reject, on every path and every method.

    Non-finite parameterisations are not listed: every float slot refuses `NaN`, `+inf` and `-inf`, and
    the sweep over `parameters` derives them. What stays here is the finite out-of-domain cases, where
    the rule is the distribution's own (`p` outside `[0, 1]`, a non-positive scale, inverted bounds).

    Arguments:
        label: Test id.
        params: The parameter tuple, in `parameters` order, so `DistSpec.build` can spell it every way.
        match: Substring of the expected `ComputeError`, naming the offending parameter. Carries enough
            of the validator's own wording to fail on the wrong parameter: a bare `"p"` would match the
            `the plugin failed with message:` preamble polars wraps every plugin error in, which would
            make the check unfailable for the four distributions whose parameter is one letter.
    """

    label: str
    params: tuple[float, ...]
    match: str


@dataclass(frozen=True)
class DistSpec:
    """One distribution.

    Arguments:
        name: Test id, and the distribution's Rust plugin prefix. Tied to `cls._distribution_name` by
            the guard, so the test id and the plugin namespace cannot drift apart.
        cls: The concrete class. `build` constructs it by keyword from `parameters`.
        continuous: `True` for a continuous distribution (has `pdf`, finite-grid integral), `False` for
            a discrete one (has `pmf`, finite-support sum).
        parameters: The constructor parameters in plugin-input order. Drives every parameter spelling,
            the constructor type contract, the non-finite sweep, and the output-naming contract (the
            *first* parameter's column is the root name a column-parameterised output inherits).
        param_strategy: Strategy yielding a valid parameter tuple, for the `hypothesis` suite.
        example: One valid parameterisation, for tests that need a fixed frame rather than a sweep.
        eval_range: `(lo, hi)` finite window for evaluating cdf / density on a grid, per drawn tuple.
        bounds: `(lo, hi)` support bounds **at `example`**, infinite where the support is unbounded.
            Outside them the density is `0`, the cdf is `0` below and `1` above, and the sf is the
            complement; `ppf(0)` / `ppf(1)` are them. Never `None`: an unbounded side is `-inf` / `inf`,
            so the assertion still runs rather than skipping.
        on_support_point: One point strictly inside the support **at `example`** where every method is well
            defined and every parameter enters the formula. The evaluation point for the
            null-propagation contract, which below the support would read a parameter-independent
            constant instead (`Bernoulli`'s `1.0` made that contract pass vacuously once).
        sample_dtype: What `sample` returns; not `Float64` for every family, since a Bernoulli draw is
            `Boolean` and a Binomial one `UInt64`. Cross-checked against the real output by the guard,
            so the record cannot drift from the plugin.
        invalid: Finite out-of-domain parameterisations, every one of which must raise.
        integration_bounds: `(lo, hi)` over which the pdf integrates to ~1 (continuous only);
            the normal is truncated to a wide multiple of `std`. `None` for discrete.
        support: Finite list of mass points summing to ~1 (discrete only). `None` for continuous.
    """

    name: DistributionName
    cls: type[_UnivariateDistribution]
    continuous: bool
    parameters: tuple[Param, ...]
    param_strategy: SearchStrategy[tuple[float, ...]]
    example: tuple[float, ...]
    eval_range: Callable[[tuple[float, ...]], tuple[float, float]]
    bounds: tuple[float, float]
    on_support_point: float
    sample_dtype: PolarsDataType
    invalid: tuple[InvalidCase, ...]
    integration_bounds: Callable[[tuple[float, ...]], tuple[float, float]] | None = None
    support: Callable[[tuple[float, ...]], list[float]] | None = None

    def build(
        self, regime: Regime, params: tuple[float, ...] | None = None, mask: pl.Expr | None = None
    ) -> _UnivariateDistribution:
        """The distribution with every parameter spelled in `regime`.

        `params` defaults to `example`. The `name` regime ignores it and reads the contract frame
        instead, since a column name carries no value of its own.
        """
        values = self.example if params is None else params
        return self.cls(
            **{p.name: _spell(regime, p, v, mask, first=i == 0) for i, (p, v) in enumerate(self.pairs(values))}
        )

    def pairs(self, params: tuple[float, ...] | None = None) -> tuple[tuple[Param, float], ...]:
        """`(parameter, value)` in plugin-input order, `params` defaulting to `example`."""
        values = self.example if params is None else params
        return tuple(zip(self.parameters, values, strict=True))

    def blamed(self, case: InvalidCase) -> Param | None:
        """The parameter `case.match` blames, read off the `"<name> must be"` wording the guard enforces.

        `None` when the wording names nothing the row declares, which
        `registry_test.test_spec_invalid_table_is_not_empty_and_names_its_parameter` refuses.
        """
        blamed = case.match.split(" must be")[0]
        return next((p for p in self.parameters if p.name == blamed), None)

    @property
    def float_parameters(self) -> tuple[Param, ...]:
        """The parameters carried as `Float64`, the ones the non-finite sweep applies to.

        `Binomial.n` and `DiscreteUniform`'s bounds are excluded: an integer column cannot hold `NaN`,
        and their dtype gate is stricter and separately asserted.
        """
        return tuple(p for p in self.parameters if not p.integer)

    @property
    def off_support_points(self) -> tuple[float, ...]:
        """Points just outside each *finite* end of the support at `example`.

        Two per finite end: one a whole unit out, and one a single ULP out. The near point is what
        pins the comparison itself. A support branch written `x < lo - 1e-8` rather than `x < lo`
        answers correctly a unit away and wrongly a ULP away, which is the regime the deleted
        `lognormal/support_test.py` probed at `-1e-9`.

        Empty only where the support is the whole line, which `support_test.py` skips explicitly
        rather than passing with an empty loop.
        """
        lo, hi = self.bounds
        below = (lo - 1.0, nextafter(lo, -inf)) if lo != -inf else ()
        above = (hi + 1.0, nextafter(hi, inf)) if hi != inf else ()
        return below + above


def _spell(regime: Regime, param: Param, value: float, mask: pl.Expr | None, *, first: bool) -> float | IntoExprColumn:
    """One parameter value in one regime. `masked` nulls the *first* parameter where `mask` is true."""
    if regime == "name":
        return param.column
    if regime == "scalar":
        return param.cast(value)
    if regime == "literal":
        return pl.lit(param.cast(value), dtype=param.dtype)
    if regime == "series":
        return pl.lit(pl.Series([param.cast(value)] * SERIES_ROWS, dtype=param.dtype))
    if regime not in ("column", "masked"):  # pragma: no cover - `Regime` is exhausted above
        msg = f"unhandled regime: {regime}"
        raise AssertionError(msg)
    column = constant_column(param.cast(value), param.dtype)
    if regime == "masked" and first:
        if mask is None:  # pragma: no cover - a caller bug, not a distribution one
            msg = "the `masked` regime needs a boolean mask expr"
            raise ValueError(msg)
        return pl.when(~mask).then(column)
    return column


_BERNOULLI = DistSpec(
    name="bernoulli",
    cls=Bernoulli,
    continuous=False,
    parameters=(Param("p", "p"),),
    param_strategy=st.tuples(_finite(0.0, 1.0)),
    example=(0.3,),
    eval_range=lambda _: (-1.0, 2.0),
    bounds=(0.0, 1.0),
    # `1.0` would be vacuous: the cdf saturates to `1` there whatever `p` is.
    on_support_point=0.0,
    sample_dtype=pl.Boolean(),
    invalid=(
        InvalidCase("p=1.5", (1.5,), "p must be in"),
        InvalidCase("p=-0.1", (-0.1,), "p must be in"),
    ),
    support=lambda _: [0.0, 1.0],
)

_BINOMIAL = DistSpec(
    name="binomial",
    cls=Binomial,
    continuous=False,
    parameters=(Param("n", "n", integer=True), Param("p", "p")),
    # `n` is drawn as an integer trial count; `p` spans the closed unit interval (incl. the
    # degenerate endpoints, where the mass collapses onto a single support point).
    param_strategy=st.tuples(st.integers(min_value=0, max_value=20), _finite(0.0, 1.0)),
    example=(7, 0.35),
    eval_range=lambda p: (-1.0, p[0] + 1.0),
    bounds=(0.0, 7.0),
    on_support_point=1.0,
    sample_dtype=pl.UInt64(),
    # A negative or oversized `n` is absent on purpose: `coerce_n` rejects it as a `ValueError` at
    # construction, so it never reaches a plugin and cannot be built through `build`. Only a column can
    # carry one to a validator, which the integer-dtype cases in `precision_test.py` cover.
    invalid=(
        InvalidCase("p=1.5", (5, 1.5), "p must be in"),
        InvalidCase("p=-1", (5, -1.0), "p must be in"),
    ),
    support=lambda p: [float(k) for k in range(int(p[0]) + 1)],
)


def _geometric_support_limit(p: float) -> int:
    """Smallest `k` whose tail mass `(1 - p)**k` falls below `1e-15`.

    The geometric support is infinite, so the suite truncates it where the missing tail sits far
    below its mass tolerance. At the degenerate `p = 1` the whole mass sits on `k = 1`."""
    return 1 if p >= 1.0 else ceil(log(1e-15) / log1p(-p))


_GEOMETRIC = DistSpec(
    name="geometric",
    cls=Geometric,
    continuous=False,
    parameters=(Param("p", "p"),),
    # `p` spans `(0, 1]`: unlike Bernoulli there is no `p = 0` point mass, since "never succeeds"
    # has no moments and statrs rejects it.
    param_strategy=st.tuples(_finite(0.05, 1.0)),
    example=(0.3,),
    eval_range=lambda p: (-1.0, float(_geometric_support_limit(p[0]) + 1)),
    bounds=(1.0, inf),
    on_support_point=1.0,
    sample_dtype=pl.UInt64(),
    invalid=(
        # The one endpoint Geometric rejects where Bernoulli accepts it: the trial count of a
        # never-succeeding trial is not representable.
        InvalidCase("p=0", (0.0,), "p must be in"),
        InvalidCase("p=-0.1", (-0.1,), "p must be in"),
        InvalidCase("p=1.5", (1.5,), "p must be in"),
    ),
    support=lambda p: [float(k) for k in range(1, _geometric_support_limit(p[0]) + 1)],
)

_DISCRETE_UNIFORM = DistSpec(
    name="discreteuniform",
    cls=DiscreteUniform,
    continuous=False,
    parameters=(Param("min", "lo_i", integer=True, paired=True), Param("max", "hi_i", integer=True, paired=True)),
    # Both bounds inclusive; the pair strategy draws independently and filters to a valid ordering.
    # Widths stay modest because several property tests evaluate every support point in `support`.
    param_strategy=st.tuples(st.integers(min_value=-15, max_value=25), st.integers(min_value=-15, max_value=25)).filter(
        lambda p: p[0] <= p[1]
    ),
    example=(1, 6),
    eval_range=lambda p: (p[0] - 1.0, p[1] + 1.0),
    bounds=(1.0, 6.0),
    on_support_point=3.0,
    sample_dtype=pl.Int64(),
    invalid=(InvalidCase("min>max", (6, 1), "max must be"),),
    support=lambda p: [float(k) for k in range(int(p[0]), int(p[1]) + 1)],
)

_NORMAL = DistSpec(
    name="normal",
    cls=Normal,
    continuous=True,
    parameters=(Param("mu", "mu"), Param("sigma", "sigma")),
    param_strategy=st.tuples(_finite(-10.0, 10.0), _finite(1e-2, 10.0)),
    example=(1.5, 2.0),
    eval_range=lambda p: (p[0] - 6.0 * p[1], p[0] + 6.0 * p[1]),
    bounds=(-inf, inf),
    # `mu` itself would be vacuous for `sigma`: the cdf is `0.5` at the median whatever the scale.
    on_support_point=1.5 + 2.0,
    sample_dtype=pl.Float64(),
    invalid=(
        InvalidCase("sigma=0", (0.0, 0.0), "sigma must be"),
        InvalidCase("sigma=-1", (0.0, -1.0), "sigma must be"),
    ),
    integration_bounds=lambda p: (p[0] - 12.0 * p[1], p[0] + 12.0 * p[1]),
)

_CAUCHY = DistSpec(
    name="cauchy",
    cls=Cauchy,
    continuous=True,
    parameters=(Param("loc", "loc"), Param("scale", "scale")),
    param_strategy=st.tuples(_finite(-10.0, 10.0), _finite(1e-2, 10.0)),
    example=(1.5, 2.0),
    eval_range=lambda p: (p[0] - 6.0 * p[1], p[0] + 6.0 * p[1]),
    bounds=(-inf, inf),
    # `loc` itself would be vacuous for `scale`: the cdf is `0.5` at the median whatever the scale.
    on_support_point=1.5 + 2.0,
    sample_dtype=pl.Float64(),
    invalid=(
        InvalidCase("scale=0", (0.0, 0.0), "scale must be"),
        InvalidCase("scale=-1", (0.0, -1.0), "scale must be"),
    ),
    # The trapezoid is spectrally accurate on a Lorentzian, so the whole error is the truncated tail
    # mass `2 / (pi k)`: `2.1e-4` at `k = 3000` against the `1e-3` tolerance, measured on the suite's grid.
    integration_bounds=lambda p: (p[0] - 3000.0 * p[1], p[0] + 3000.0 * p[1]),
)

_UNIFORM = DistSpec(
    name="uniform",
    cls=Uniform,
    continuous=True,
    parameters=(Param("min", "lo", paired=True), Param("max", "hi", paired=True)),
    # Draw `(min, width)` with `width > 0`, then expose `(min, max)`; this guarantees `max > min`
    # without rejection, so every drawn parameterisation is valid.
    param_strategy=st.tuples(_finite(-10.0, 10.0), _finite(1e-2, 20.0)).map(lambda mw: (mw[0], mw[0] + mw[1])),
    example=(-1.0, 3.0),
    eval_range=lambda p: (p[0] - 0.5 * (p[1] - p[0]), p[1] + 0.5 * (p[1] - p[0])),
    bounds=(-1.0, 3.0),
    on_support_point=1.0,
    sample_dtype=pl.Float64(),
    invalid=(
        InvalidCase("max<min", (2.0, 1.0), "max must be"),
        InvalidCase("max=min", (1.0, 1.0), "max must be"),
        # Individually finite bounds whose width overflows `f64`: uniform's own guard, not statrs'.
        InvalidCase("width overflows", (-1e308, 1e308), "max must be"),
    ),
    integration_bounds=lambda p: (p[0], p[1]),
)

_LOGNORMAL = DistSpec(
    name="lognormal",
    cls=LogNormal,
    continuous=True,
    parameters=(Param("mu", "mu"), Param("sigma", "sigma")),
    # `sigma` is capped at 0.9: the heavy right tail makes a uniform-grid trapezoidal integral lose
    # accuracy as `sigma` grows, and past ~1.2 the mass check drifts above the 1e-3 tolerance at the
    # current `_INTEGRATION_GRID_SIZE`. The functional and scipy-parity suites cover larger `sigma`.
    param_strategy=st.tuples(_finite(-1.5, 1.5), _finite(0.1, 0.9)),
    example=(0.5, 0.75),
    # Support is (0, inf); the grid stays on the positive side and out to a 4-sigma-in-log upper tail.
    eval_range=lambda p: (0.0, exp(p[0] + 4.0 * p[1])),
    bounds=(0.0, inf),
    # One log-sigma above the median `exp(mu)`, which on its own would read `0.5` for any `sigma`.
    on_support_point=exp(0.5 + 0.75),
    sample_dtype=pl.Float64(),
    invalid=(InvalidCase("sigma=-1", (0.0, -1.0), "sigma must be"),),
    integration_bounds=lambda p: (0.0, exp(p[0] + 6.0 * p[1])),
)

_EXPONENTIAL = DistSpec(
    name="exponential",
    cls=Exponential,
    continuous=True,
    parameters=(Param("rate", "rate"),),
    # `rate > 0`; the lower bound keeps the mean `1 / rate` finite enough for the trapezoidal mass
    # check, the upper bound keeps it from collapsing onto a near-degenerate spike at 0.
    param_strategy=st.tuples(_finite(1e-2, 10.0)),
    example=(1.5,),
    # Support is [0, inf); the grid spans the `x < 0` zero region through several means.
    eval_range=lambda p: (-1.0 / p[0], 5.0 / p[0]),
    bounds=(0.0, inf),
    on_support_point=1.0 / 1.5,
    sample_dtype=pl.Float64(),
    invalid=(
        InvalidCase("rate=0", (0.0,), "rate must be"),
        InvalidCase("rate=-1", (-1.0,), "rate must be"),
    ),
    # `1 - exp(-30)` of the mass lies below `30 / rate`; the rest is below the 1e-3 tolerance.
    integration_bounds=lambda p: (0.0, 30.0 / p[0]),
)

_PARETO = DistSpec(
    name="pareto",
    cls=Pareto,
    continuous=True,
    parameters=(Param("scale", "scale"), Param("shape", "shape")),
    # Below `shape ~ 1.1` the suite's uniform grid stops resolving the density near `scale`. The
    # divergent-moment shapes (`<= 1` for the mean, `<= 2` for the variance and the standard
    # deviation) are covered by the parity, moments and precision suites instead.
    param_strategy=st.tuples(_finite(0.1, 10.0), _finite(1.5, 10.0)),
    example=(1.5, 3.0),
    # From the zero region below `scale` out to `sf = 0.01`.
    eval_range=lambda p: (0.5 * p[0], p[0] * 100.0 ** (1.0 / p[1])),
    bounds=(1.5, inf),
    # Not `scale`: the cdf is `0` there whatever the tail index.
    on_support_point=3.0,
    sample_dtype=pl.Float64(),
    invalid=(
        InvalidCase("scale=0", (0.0, 3.0), "scale must be"),
        InvalidCase("scale=-1", (-1.0, 3.0), "scale must be"),
        InvalidCase("shape=0", (1.5, 0.0), "shape must be"),
        InvalidCase("shape=-1", (1.5, -1.0), "shape must be"),
    ),
    # Out to `sf = 1 / 2000`, half the mass tolerance; the trapezoid is good to `3e-5` at `shape = 1.5`.
    integration_bounds=lambda p: (p[0], p[0] * 2000.0 ** (1.0 / p[1])),
)

_BETA = DistSpec(
    name="beta",
    cls=Beta,
    continuous=True,
    parameters=(Param("a", "a"), Param("b", "b")),
    # Shapes are kept >= 1 so the density is finite at the support endpoints: below 1 it diverges
    # there, making `pdf(0)` or `pdf(1)` inf and the trapezoidal mass integer inf with it. Finite
    # is not the same as accurately integrable: just above 1 the density stays high right up to the
    # endpoint and the rule drops the final half-cell, which is what sizes `_INTEGRATION_GRID_SIZE`.
    # Shapes below 1 are covered directly by the functional and scipy-parity suites.
    param_strategy=st.tuples(_finite(1.0, 10.0), _finite(1.0, 10.0)),
    example=(2.0, 3.0),
    # Support is [0, 1]; the grid spans the zero-density regions on both sides.
    eval_range=lambda _: (-0.5, 1.5),
    bounds=(0.0, 1.0),
    on_support_point=0.5,
    sample_dtype=pl.Float64(),
    invalid=(
        InvalidCase("a=0", (0.0, 1.0), "a must be"),
        InvalidCase("b=-1", (2.0, -1.0), "b must be"),
    ),
    integration_bounds=lambda _: (0.0, 1.0),
)

NON_FINITE = {"NaN": float("nan"), "inf": inf, "-inf": -inf}
"""The three non-finite values every float parameter slot must refuse, keyed by the spelling the
validators echo back in `got <value>`. That suffix is what makes the derived match discriminating:
`Uniform`'s pair rule reads `max - min must be finite`, so a bare `"min must be"` is a substring of
the *sibling's* refusal and would pass with `min`'s own finiteness check deleted."""


def invalid_cases(specs: list[DistSpec]) -> list[tuple[DistSpec, InvalidCase]]:
    """Every parameterisation every path must refuse, for each spec in `specs`.

    The recorded finite out-of-domain rows first, then the non-finite sweep derived from
    `float_parameters`: each slot set to `NaN`, `+inf` and `-inf` in turn with its siblings valid. One
    source for the three tables that used to hold this and had drifted apart.
    """
    cases: list[tuple[DistSpec, InvalidCase]] = [(spec, case) for spec in specs for case in spec.invalid]
    cases += [
        (
            spec,
            InvalidCase(
                f"{slot.name}={label}",
                tuple(bad if p is slot else v for p, v in spec.pairs()),
                rf"{slot.name} must be .*, got {label}",
            ),
        )
        for spec in specs
        for slot in spec.float_parameters
        for label, bad in NON_FINITE.items()
    ]
    return cases


CONTRACT_FRAME = pl.DataFrame(
    {
        # One column per distinct parameter across every distribution, valid on every row. Parameters
        # within a distribution never share a column, which is what makes the output-naming assertions
        # discriminating: an expression that followed `inputs[1]` rather than polars' first-input rule
        # would inherit the wrong name and fail. `lo_i` / `hi_i` are the discrete uniform's own bounds
        # rather than a reuse of `n`, for the same reason.
        "mu": [0.0, 1.0, -0.5],
        "sigma": [1.0, 2.0, 0.5],
        "loc": [0.0, 1.0, -0.5],
        "scale": [1.0, 2.0, 0.5],
        "shape": [2.5, 3.0, 1.5],
        "lo": [0.0, -1.0, 2.0],
        "hi": [1.0, 3.0, 5.0],
        "lo_i": [0, -3, 2],
        "hi_i": [5, 1, 2],
        "n": [5, 10, 20],
        "p": [0.2, 0.5, 0.9],
        "rate": [1.0, 2.0, 0.5],
        "a": [2.0, 1.0, 3.0],
        "b": [3.0, 1.0, 0.5],
        "x": [0.5, 1.5, 3.0],  # evaluation points for pdf / pmf / cdf / sf
        "q": [0.1, 0.5, 0.9],  # quantiles in [0, 1] for ppf / isf
    },
    schema_overrides={"n": pl.Int64, "lo_i": pl.Int64, "hi_i": pl.Int64},
)
"""The frame the `name` parameter regime reads. Its schema is the contract `Param.column` names into."""


ALL_SPECS = [
    _BERNOULLI,
    _BINOMIAL,
    _DISCRETE_UNIFORM,
    _GEOMETRIC,
    _NORMAL,
    _CAUCHY,
    _UNIFORM,
    _LOGNORMAL,
    _EXPONENTIAL,
    _PARETO,
    _BETA,
]
SPECS_BY_NAME: dict[DistributionName, DistSpec] = {spec.name: spec for spec in ALL_SPECS}
CONTINUOUS_SPECS = [s for s in ALL_SPECS if s.continuous]
DISCRETE_SPECS = [s for s in ALL_SPECS if not s.continuous]

# These specs compute a moment in Polars, not in a Rust body, so with constant parameters their
# operands are length-1 literals that polars may fold differently from the column kernel. Every IEEE
# operation is exactly rounded, so the measured 1-ULP gap (Uniform's `variance` / `std`) is a
# different operation *order*, not a different formula. Extend from a failing assertion, never by
# widening the tolerance. There is no value-keyed counterpart: every value-keyed method runs in Rust,
# so one body feeds both routings and they are bit-exact by construction.
ULP_TOLERANT_MOMENTS: dict[DistributionName, frozenset[Moment]] = {
    "uniform": frozenset({"variance", "std"}),
    "geometric": frozenset({"std", "entropy"}),
}
"""The `(spec, moment)` pairs that compare to `ULP_REL_TOL` instead of bit-exactly.

Keyed by moment, not by spec: `uniform`'s `mean` and `median` and `geometric`'s `mean` and `variance`
are bit-exact, and keying the whole spec silently relaxed them too.

`geometric`'s two are here for a narrower reason than `uniform`'s: both divide by the parameter
itself, and `col`'s `pl.repeat` keeps `p` in polars' scalar-backed representation, whose division
kernel is a reciprocal multiply rather than an exactly-rounded divide. A materialised `p` column and
a `pl.lit(pl.Series(...))` one both stay bit-exact against the fast path (0 divergences over 300
random `p`); only the `pl.repeat` spelling moves the last bit.
"""


def compares_bit_exactly(spec_name: DistributionName, moment: Moment) -> bool:
    """Whether this spec's `moment` must agree to the last bit across the two parameter routings."""
    return moment not in ULP_TOLERANT_MOMENTS.get(spec_name, frozenset())


UNDEFINED_MOMENTS: dict[DistributionName, frozenset[Moment]] = {"cauchy": frozenset({"mean", "variance", "std"})}
"""The `(spec, moment)` pairs with no value: **null** on every valid row, on both parameter routings, and still
raising on an invalid parameterisation.

`docs/explanation/design.md` settles the contract. A *divergent* moment is `+inf`, which is a value, and is not
listed here.
"""


def moment_is_undefined(spec_name: DistributionName, moment: Moment) -> bool:
    """Whether this spec's `moment` is recorded as having no value."""
    return moment in UNDEFINED_MOMENTS.get(spec_name, frozenset())


# Parameterisations where the mass collapses onto one point: `p` at an endpoint, no trials, a
# one-point range. Both routings must accept them, agree, and reach a finite entropy by the
# `0 log 0 = 0` convention rather than raising.
#
# Not a `DistSpec` field: seven of the eleven distributions have none, and a field that is an empty
# tuple more often than not is a bespoke fact wearing a row's clothes. It lives here rather than in
# `fast_path_test.py` because `registry_test.py` reads it too.
DEGENERATE: dict[str, tuple[DistributionName, tuple[float, ...]]] = {
    "bernoulli p=0": ("bernoulli", (0.0,)),
    "bernoulli p=1": ("bernoulli", (1.0,)),
    "geometric p=1": ("geometric", (1.0,)),
    "binomial n=0": ("binomial", (0, 0.5)),
    "binomial p=0": ("binomial", (5, 0.0)),
    "binomial p=1": ("binomial", (5, 1.0)),
    "discreteuniform min=max": ("discreteuniform", (3, 3)),
}


ULP_REL_TOL = 1e-15
"""~4x a double's ULP."""

ULP_ABS_TOL = 0.0
"""No absolute slack, so a tiny unequal pair cannot pass for free."""
