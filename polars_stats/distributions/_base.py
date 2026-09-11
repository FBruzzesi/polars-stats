from __future__ import annotations

import inspect
import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, TypeGuard

import polars as pl
from polars.exceptions import PolarsError
from polars.plugins import register_plugin_function

from polars_stats._lib import LIB

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from typing_extensions import TypeIs

    from polars_stats._typing import (
        DistributionName,
        IntoExprColumn,
        ParamFunction,
        PluginFunction,
        PolarsDataType,
        SamplerFunction,
        ValueFunction,
    )

_LITERAL_LEN_IN_AGG = tuple(int(part) for part in pl.__version__.split(".", 2)[:2]) >= (1, 35)
"""Whether `pl.lit(...).len()` survives inside `over` / `group_by().agg()`."""

_ROW_INDEX_NAME = "__polars_stats_row_index__"
ROW_INDEX_EXPR = pl.int_range(0, pl.len(), dtype=pl.UInt64).alias(_ROW_INDEX_NAME)
"""Per-row position `0..len`, the per-row sub-seed of the constant-parameter samplers.

`pl.len()` is the frame length under `select` / `with_columns` and the partition length under `over` /
`group_by`. Only the `*_scalar` samplers use it as is; the per-row samplers take `row_index_expr`.
"""


def _frame_free_length(param: pl.Expr) -> int | None:
    """The parameter's length if it can be evaluated without a frame, else `None`."""
    try:
        return pl.select(param).height
    except PolarsError:
        return None


def row_index_expr(params: Iterable[pl.Expr]) -> pl.Expr:
    """`ROW_INDEX_EXPR` sized by the call's row count instead of the frame height.

    A parameter longer than the frame sets the row count. A `pl.len()`-sized index would then be
    length 1, polars would broadcast it, and every row would seed from position 0. The plugin cannot
    repair that: the streaming engine splits such a call into one-row morsels.

    Length-1 parameters never set the row count, as in `align_inputs`. Each parameter's `.len()` is
    mentioned once: polars evaluates a repeated subexpression once per mention. Below polars 1.35 a
    literal's length cannot be asked for inside a partition context, so the frame-free lengths are
    resolved here by `pl.select` and only the frame-dependent ones are left to `.len()`.
    """
    sized = (param for param in params if not param.meta.is_column())
    spans: list[int | pl.Expr]
    if _LITERAL_LEN_IN_AGG:
        spans = [param.len().replace(1, 0) for param in sized]
    else:
        spans, fixed = [], 0
        for param in sized:
            if (length := _frame_free_length(param)) is None:
                spans.append(param.len().replace(1, 0))
            else:
                fixed = max(fixed, 0 if length == 1 else length)
        if fixed:
            spans.insert(0, fixed)

    if not spans:
        return ROW_INDEX_EXPR
    return pl.int_range(0, pl.max_horizontal(pl.len(), *spans), dtype=pl.UInt64).alias(_ROW_INDEX_NAME)


def _is_number(value: object) -> TypeGuard[int | float]:
    """A plain `int` or `float`; `bool` is neither, though it subclasses `int`."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int(value: object) -> TypeGuard[int]:
    return _is_number(value) and isinstance(value, int)


def _coerce(
    value: float | IntoExprColumn,
    *,
    name: str,
    scalar_label: str,
    scalar_types: type | tuple[type, ...],
    dtype: PolarsDataType | None = None,
) -> pl.Expr:
    """Coerce a scalar or `IntoExprColumn` input into a `pl.Expr`.

    A `pl.Expr` passes through, a `str` becomes `pl.col(name)`, a `pl.Series` becomes `pl.lit(series)`,
    and a Python scalar of `scalar_types` becomes `pl.lit(value, dtype)`. `bool` is always rejected.

    A constant stays length 1: row alignment belongs to the plugin (`align_inputs`), so an expression
    built only from constants is a scalar column with polars' own semantics (height 1 under `select`,
    broadcast per partition under `over`, a scalar per group under `group_by().agg()`).
    """
    if isinstance(value, pl.Expr):
        return value
    if isinstance(value, str):
        return pl.col(value)
    if isinstance(value, pl.Series):
        return pl.lit(value)
    if _is_number(value) and isinstance(value, scalar_types):
        return pl.lit(value, dtype=dtype)
    msg = f"{name} should be {scalar_label} or IntoExprColumn (pl.Expr, str, pl.Series), found {type(value)}"
    raise TypeError(msg)


def coerce_param(value: float | IntoExprColumn, *, name: str) -> pl.Expr:
    """Coerce a float parameter (`mu`, `sigma`, `p`, ...). An `int` or `bool` scalar raises `TypeError`."""
    return _coerce(value, name=name, scalar_label="a float", scalar_types=float, dtype=pl.Float64())


_MAX_WIRE_INT = 2**63 - 1
"""Largest integer that reaches a plugin: kwargs cross FFI as pickle, whose integers are `i64`."""


def coerce_n(value: int | IntoExprColumn, *, name: str) -> pl.Expr:
    """Coerce a count parameter into a `UInt64` literal or a column the plugin widens to `UInt64`.

    The one parameter whose *value* is judged at construction: a Python `int` outside
    `[0, _MAX_WIRE_INT]` raises `ValueError`, since neither the `UInt64` literal nor the kwargs wire can
    carry it. A column keeps its dtype until Rust widens it.
    """
    if (trials := scalar_int(value)) is not None:
        if trials < 0:
            msg = f"{name} must be a non-negative integer, got {trials}"
            raise ValueError(msg)
        if trials > _MAX_WIRE_INT:
            msg = f"{name} must be at most {_MAX_WIRE_INT} as a Python int, got {trials}: pass a column instead"
            raise ValueError(msg)
    return _coerce(value, name=name, scalar_label="an int", scalar_types=int, dtype=pl.UInt64())


def coerce_int(value: int | IntoExprColumn, *, name: str) -> pl.Expr:
    """Coerce a signed integer parameter into an `Int64` literal or a column the plugin widens to `Int64`.

    A Python `int` outside `Int64` raises `ValueError` at construction, as `coerce_n` does.
    """
    if (bound := scalar_int(value)) is not None and not -(2**63) <= bound <= _MAX_WIRE_INT:
        msg = f"{name} must be in [{-(2**63)}, {_MAX_WIRE_INT}] as a Python int, got {bound}: pass a column instead"
        raise ValueError(msg)
    return _coerce(value, name=name, scalar_label="an int", scalar_types=int, dtype=pl.Int64())


def scalar_float(value: float | IntoExprColumn) -> float | None:
    """`value` as a `float` if it is a plain numeric scalar, else `None`."""
    return float(value) if _is_number(value) else None


def scalar_int(value: int | IntoExprColumn) -> int | None:
    """`value` if it is a plain `int` (not a `bool`), else `None`."""
    return value if _is_int(value) else None


def scalar_kwargs(**params: float | None) -> dict[str, float | int] | None:
    """The constant-parameter kwargs when every parameter is a scalar, else `None`.

    Each keyword has gone through `scalar_float` / `scalar_int`, so `None` marks a column-valued
    parameter, and one column collapses the whole bundle to `None`.
    """
    return (
        None if any(value is None for value in params.values()) else {k: v for k, v in params.items() if v is not None}
    )


def as_expr(value: float | IntoExprColumn) -> pl.Expr:
    """Coerce a value-keyed method input (`value` / `quantile`); an `int` or `float` scalar is accepted."""
    return _coerce(value, name="value", scalar_label="a number (int or float)", scalar_types=(int, float))


def _checked_int(value: object, *, name: str, expected: str = "an int") -> int:
    if _is_int(value):
        return value
    msg = f"{name} should be {expected}, found {type(value)}"
    raise TypeError(msg)


def _checked_seed(seed: object) -> int | None:
    """A sampler `seed` the kwargs wire can carry, or `None` for OS entropy."""
    if seed is None:
        return None
    if not 0 <= (seed := _checked_int(seed, name="seed", expected="an int or None")) <= _MAX_WIRE_INT:
        msg = f"seed must be in [0, 2**63), got {seed}"
        raise ValueError(msg)
    return seed


def _checked_size(size: object) -> int:
    """A positive draw count. No maximum: an oversized one dies in the allocator, not here."""
    if (size := _checked_int(size, name="size")) <= 0:
        msg = f"size must be a positive integer, got {size}"
        raise ValueError(msg)
    return size


def register_plugin(
    distribution: DistributionName,
    function: PluginFunction,
    args: IntoExprColumn | Iterable[IntoExprColumn],
    *,
    kwargs: Mapping[str, float | int | None] | None = None,
    scalar: bool = False,
) -> pl.Expr:
    """Register the Rust plugin `<distribution>_<function>`, or its `_scalar` twin when `scalar` is set.

    `is_elementwise=True` is fixed: every distribution plugin is per-row by contract, and an aggregating
    one would break `over` / `group_by`. `kwargs` carries only what cannot be column-valued: a sampler
    `seed`, a draw count `size`, and the constant parameters of a `_scalar` twin.
    """
    return register_plugin_function(
        plugin_path=LIB,
        function_name=f"{distribution}_{function}_scalar" if scalar else f"{distribution}_{function}",
        args=args,
        kwargs=None if kwargs is None else dict(kwargs),
        is_elementwise=True,
    )


def expm1(t: pl.Expr) -> pl.Expr:
    """`exp(t) - 1` as `2 exp(t / 2) sinh(t / 2)`, which has no cancelling subtraction.

    Polars exposes no `expm1`, and the literal form cancels to `0` below `|t| ~ 1.1e-16`. `sinh(t / 2)`
    overflows above `|t| ~ 1420`; for `t > 0` the true answer overflows first, for `t < 0` callers that
    reach the far tail pick their own crossover.
    """
    half = t / 2
    return 2 * half.exp() * half.sinh()


def log_abs_expm1(t: pl.Expr) -> pl.Expr:
    """`log|exp(t) - 1|` as `log(2) + t / 2 + log|sinh(t / 2)|`, `expm1`'s identity read on the log scale.

    Each term is large exactly where the answer is large, so rounding stays relative where
    `expm1(t).log()` would round the answer's own magnitude away. For `t < 0` this is the log of the
    complement `1 - exp(t)`.
    """
    half = t / 2
    return math.log(2.0) + half + half.sinh().abs().log()


class _UnivariateDistribution(ABC):
    """A univariate probability distribution whose methods return polars expressions.

    Parameters may be Python scalars or `pl.Expr`, so one instance can describe a different
    distribution per row (`Normal(mu=pl.col("mu"), sigma=1.0)`). The interface mirrors
    `scipy.stats.rv_continuous` / `rv_discrete`.

    A public value-keyed method only coerces its argument; the `_x` hook it calls answers every row,
    null and `NaN` rows included. Every hook defaults to the Rust plugin `<name>_<method>`, so a
    subclass declares its parameters, its validator and its closed-form moments and nothing else.
    """

    _distribution_name: ClassVar[DistributionName]
    """Prefix of the distribution's Rust plugins: every one it calls is `<name>_<function>`."""

    _scalar_kwargs: dict[str, float | int] | None
    """Constant parameters for the `_scalar` fast paths, `None` when any parameter is column-valued.

    Set by each subclass at construction via `scalar_kwargs`, in `_param_exprs` order. Also the
    routing switch of every plugin call and the naming switch of the samplers: with no input column to
    inherit a name from, `sample` / `samples` are named after themselves.
    """

    @property
    @abstractmethod
    def _param_exprs(self) -> tuple[pl.Expr, ...]:
        """The coerced parameters in plugin-input order.

        The order is the contract: the Rust side reads the inputs positionally, and the output is
        named after the first expression (polars root-name semantics, pinned by `output_name_test.py`).
        """

    @property
    @abstractmethod
    def _validated_params(self) -> pl.Expr:
        """The validating plugin call the moments gate on.

        A Rust plugin expr that raises on an invalid parameterisation and is null when any parameter
        is null; every validator returns non-null only when all its parameters are. Built with
        `_validated` (a reused quantity) or `_param_plugin` (the plugin's own output).
        """

    def __repr__(self) -> str:
        """One line shaped like the constructor call, e.g. `Normal(mu=0.0, sigma=col("s"))`.

        Constant parameters render from `_scalar_kwargs`, so the values stay exact. Column-valued ones
        render as `str` of their `_param_exprs` entry under the `__init__` parameter names.
        """
        cls = type(self)
        if (scalars := self._scalar_kwargs) is not None:
            params = ", ".join(f"{name}={value}" for name, value in scalars.items())
        else:
            _self, *names = inspect.signature(cls.__init__).parameters
            params = ", ".join(f"{name}={expr!s}" for name, expr in zip(names, self._param_exprs, strict=True))
        return f"{cls.__name__}({params})"

    def sample(self, seed: int | None = None) -> pl.Expr:
        """Draw one random variate per row.

        Output length follows the surrounding context (frame length under `select` / `with_columns`,
        partition length under `over` / `group_by`). Each row's draw derives from a sub-seed mixed from
        `seed` and the row's position, so the result is independent of chunking and thread scheduling.

        A row with an invalid parameter raises; a row with a null parameter yields null. The output is
        named `"sample"` when every parameter is constant; otherwise it follows the first parameter
        expression (polars root-name semantics, so `.name.*` modifiers keep working).
        """
        return self._draw("sample", seed=_checked_seed(seed))

    def samples(self, size: int, seed: int | None = None) -> pl.Expr:
        """Draw `size` random variates per row, as `Array(inner=<element dtype>, shape=size)`.

        Each row's draws are consecutive values from one per-row stream keyed by `seed` and the row's
        position, so `samples(size=1)` matches `sample` for the same seed and growing `size` extends
        each row's array without changing the existing draws. A null-parameter row yields a null array
        (outer validity), an invalid parameterisation raises. Named like `sample`, as `"samples"`.
        """
        return self._draw("samples", size=_checked_size(size), seed=_checked_seed(seed))

    def _draw(self, function: SamplerFunction, **kwargs: int | None) -> pl.Expr:
        """Route a sampler: constant parameters ride in kwargs of the `_scalar` twin, columns cross FFI per row.

        Both twins seed by position and share one `draw`, so their output is bit-identical (pinned by
        `tests/property/sample_test.py`).
        """
        if self._scalar_kwargs is not None:
            return register_plugin(
                self._distribution_name,
                function,
                (ROW_INDEX_EXPR,),
                kwargs={**kwargs, **self._scalar_kwargs},
                scalar=True,
            ).alias(function)
        params = self._param_exprs
        return register_plugin(self._distribution_name, function, (*params, row_index_expr(params)), kwargs=kwargs)

    def _value_plugin(self, function: ValueFunction, value: pl.Expr) -> pl.Expr:
        """The value-keyed plugin `<name>_<function>(value, *params)`, or its `_scalar` twin.

        Parameters are validated inside the plugin, so every value-keyed method reports an invalid
        parameterisation and propagates nulls per row. The twin validates once and takes only `value`
        across FFI; both call one Rust body, so they are bit-identical.
        """
        if self._scalar_kwargs is not None:
            return register_plugin(self._distribution_name, function, (value,), kwargs=self._scalar_kwargs, scalar=True)
        return register_plugin(self._distribution_name, function, (value, *self._param_exprs))

    @property
    def _param_lits(self) -> list[pl.Expr]:
        """The constant parameters as length-1 literals, in `_param_exprs` order, so a plugin runs once."""
        assert self._scalar_kwargs is not None  # noqa: S101  # guarded by every caller
        return [pl.lit(value) for value in self._scalar_kwargs.values()]

    def _param_plugin(self, function: ParamFunction) -> pl.Expr:
        """The parameter-keyed plugin `<name>_<function>(*params)`: a validator, or a moment with no closed form.

        Column parameters run it per row; constant parameters run it once on length-1 literals, so
        the result is a scalar column. There is no `_scalar` twin.
        """
        args = self._param_exprs if self._scalar_kwargs is None else self._param_lits
        return register_plugin(self._distribution_name, function, args)

    def _validated(self, function: ParamFunction, quantity: pl.Expr) -> pl.Expr:
        """`quantity` behind the validating plugin `<name>_<function>`.

        With column parameters the plugin runs per row and its own output is returned. With constant
        parameters it runs once on length-1 literals and `quantity` (the same value recomputed in
        polars, e.g. `self._sigma`) is returned behind that gate, so the expression stays a scalar
        column. Both paths raise on the same parameters; they agree bit for bit except where polars
        folds the literal arithmetic with a different kernel than the length-n column.
        """
        if self._scalar_kwargs is None:
            return self._param_plugin(function)
        return pl.when(self._param_plugin(function).is_not_null()).then(quantity)

    def _moment(self, formula: pl.Expr) -> pl.Expr:
        """`formula` gated on `_validated_params`, so it raises on an invalid parameterisation and nulls on a null one.

        `formula` reads the raw parameter exprs: the gate is the one validator mention, so a moment
        that names a parameter several times still crosses FFI once.
        """
        return pl.when(self._validated_params.is_not_null()).then(formula)

    def cdf(self, value: float | IntoExprColumn) -> pl.Expr:
        """Cumulative distribution function, `P(X <= value)`. Nulls and NaNs in `value` are propagated."""
        return self._cdf(as_expr(value))

    def _cdf(self, value: pl.Expr) -> pl.Expr:
        return self._value_plugin("cdf", value)

    def log_cdf(self, value: float | IntoExprColumn) -> pl.Expr:
        """Natural logarithm of the cdf. Nulls and NaNs in `value` are propagated."""
        return self._log_cdf(as_expr(value))

    def _log_cdf(self, value: pl.Expr) -> pl.Expr:
        """`<name>_ln_cdf`, a stable form; a distribution without one overrides with `self._cdf(value).log()`."""
        return self._value_plugin("ln_cdf", value)

    def sf(self, value: float | IntoExprColumn) -> pl.Expr:
        """Survival function, `P(X > value) = 1 - cdf(value)`. Nulls and NaNs in `value` are propagated."""
        return self._sf(as_expr(value))

    def _sf(self, value: pl.Expr) -> pl.Expr:
        return self._value_plugin("sf", value)

    def log_sf(self, value: float | IntoExprColumn) -> pl.Expr:
        """Natural logarithm of the survival function. Nulls and NaNs in `value` are propagated."""
        return self._log_sf(as_expr(value))

    def _log_sf(self, value: pl.Expr) -> pl.Expr:
        return self._value_plugin("ln_sf", value)

    def ppf(self, quantile: float | IntoExprColumn) -> pl.Expr:
        """Percent point function (inverse cdf).

        A `quantile` outside `[0, 1]` yields **null**. Nulls are propagated and a `NaN` quantile yields
        `NaN`, matching scipy.
        """
        return self._ppf(as_expr(quantile))

    def _ppf(self, quantile: pl.Expr) -> pl.Expr:
        return self._value_plugin("ppf", quantile)

    def isf(self, quantile: float | IntoExprColumn) -> pl.Expr:
        """Inverse survival function, the value `x` with `sf(x) == quantile`.

        Same domain contract as `ppf`, with the endpoints reversed: `quantile` outside `[0, 1]` yields
        null, nulls propagate, `NaN` yields `NaN`.
        """
        return self._isf(as_expr(quantile))

    def _isf(self, quantile: pl.Expr) -> pl.Expr:
        """`<name>_isf`, solved against `quantile` itself: `ppf(1 - quantile)` quantises a small quantile first."""
        return self._value_plugin("isf", quantile)

    @abstractmethod
    def mean(self) -> pl.Expr:
        """Expected value `E[X]`."""

    @abstractmethod
    def variance(self) -> pl.Expr:
        """Variance `Var[X] = E[(X - E[X])^2]`."""

    def std(self) -> pl.Expr:
        """Standard deviation, `sqrt(variance)`."""
        return self.variance().sqrt()

    def median(self) -> pl.Expr:
        """Median, `ppf(0.5)`."""
        return self._ppf(as_expr(0.5))

    @abstractmethod
    def entropy(self) -> pl.Expr:
        """Differential or Shannon entropy, in nats."""


class DiscreteDistribution(_UnivariateDistribution, ABC):
    """Abstract base class for discrete univariate distributions."""

    def pmf(self, value: float | IntoExprColumn) -> pl.Expr:
        """Probability mass function, `P(X = value)`. Nulls and NaNs in `value` are propagated."""
        return self._pmf(as_expr(value))

    def _pmf(self, value: pl.Expr) -> pl.Expr:
        return self._value_plugin("pmf", value)

    def log_pmf(self, value: float | IntoExprColumn) -> pl.Expr:
        """Natural logarithm of the pmf. Nulls and NaNs in `value` are propagated."""
        return self._log_pmf(as_expr(value))

    def _log_pmf(self, value: pl.Expr) -> pl.Expr:
        return self._value_plugin("ln_pmf", value)


class ContinuousDistribution(_UnivariateDistribution, ABC):
    """Abstract base class for continuous univariate distributions."""

    def pdf(self, value: float | IntoExprColumn) -> pl.Expr:
        """Probability density function evaluated at `value`. Nulls and NaNs in `value` are propagated."""
        return self._pdf(as_expr(value))

    def _pdf(self, value: pl.Expr) -> pl.Expr:
        return self._value_plugin("pdf", value)

    def log_pdf(self, value: float | IntoExprColumn) -> pl.Expr:
        """Natural logarithm of the pdf. Nulls and NaNs in `value` are propagated."""
        return self._log_pdf(as_expr(value))

    def _log_pdf(self, value: pl.Expr) -> pl.Expr:
        return self._value_plugin("ln_pdf", value)


def is_discrete(obj: object, /) -> TypeIs[DiscreteDistribution]:
    """Whether ``obj`` is a discrete distribution.

    Narrowing guard: on the true branch a type checker sees ``obj`` as a
    ``DiscreteDistribution``, so ``pmf`` / ``log_pmf`` are reachable without a cast, and on the
    false branch it drops ``DiscreteDistribution`` from the type.

    Arguments:
        obj: Any object.

    Returns:
        ``True`` if ``obj`` is an instance of ``DiscreteDistribution``.

    Examples:
        >>> import polars_stats as ps
        >>> ps.is_discrete(ps.Binomial(10, 0.5))
        True
        >>> ps.is_discrete(ps.Normal())
        False

        Dispatching on the kind picks the density method that exists:

        >>> import polars as pl
        >>> def density(dist: ps.DiscreteDistribution | ps.ContinuousDistribution, value: str):
        ...     return dist.pmf(value) if ps.is_discrete(dist) else dist.pdf(value)
        >>> pl.DataFrame({"x": [0.0, 1.0]}).select(density(ps.Binomial(10, 0.5), "x"))
        shape: (2, 1)
        ┌──────────┐
        │ x        │
        │ ---      │
        │ f64      │
        ╞══════════╡
        │ 0.000977 │
        │ 0.009766 │
        └──────────┘
    """
    return isinstance(obj, DiscreteDistribution)


def is_continuous(obj: object, /) -> TypeIs[ContinuousDistribution]:
    """Whether ``obj`` is a continuous distribution.

    The counterpart of [`is_discrete`][polars_stats.is_discrete]: the two kinds are disjoint, so
    exactly one of the guards holds for any distribution in this package.

    Arguments:
        obj: Any object.

    Returns:
        ``True`` if ``obj`` is an instance of ``ContinuousDistribution``.

    Examples:
        >>> import polars_stats as ps
        >>> ps.is_continuous(ps.Normal())
        True
        >>> ps.is_continuous(ps.Binomial(10, 0.5))
        False

        The guard narrows, so ``pdf`` type checks inside the branch:

        >>> import polars as pl
        >>> dist = ps.Normal(mu=0.0, sigma=1.0)
        >>> if ps.is_continuous(dist):
        ...     print(round(pl.DataFrame({"x": [0.0]}).select(dist.pdf("x")).item(), 6))
        0.398942
    """
    return isinstance(obj, ContinuousDistribution)
