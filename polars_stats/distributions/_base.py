from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import polars as pl
from polars.plugins import register_plugin_function

from polars_stats._lib import LIB

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from polars_stats._typing import IntoExprColumn, PolarsDataType

_ROW_INDEX_NAME = "__polars_stats_row_index__"
ROW_INDEX_EXPR = pl.int_range(0, pl.len(), dtype=pl.UInt64).alias(_ROW_INDEX_NAME)
"""Per-row position `0..len` used to derive per-row sub-seeds in samplers.

Evaluated inside the surrounding context, so `pl.len()` is the frame length under
`select` / `with_columns` and the partition length under `over` / `group_by`.
"""


def _coerce(
    value: float | IntoExprColumn,
    *,
    name: str,
    scalar_label: str,
    scalar_types: type | tuple[type, ...],
    dtype: PolarsDataType | None = None,
) -> pl.Expr:
    """Coerce a scalar-or-`IntoExprColumn` input into a row-aligned `pl.Expr`.

    Every distribution input (constructor parameter or value-keyed method argument) is one of two things, handled
    identically here so the contract lives in one place:

    * An `IntoExprColumn`: a `pl.Expr` passes through; a column name `str` becomes `pl.col(name)`, a `pl.Series`
        becomes `pl.lit(series)`.
    * A Python scalar: expanded with `pl.repeat(value, n=pl.len())` rather than `pl.lit(value)`.

    NOTE: The scalar expansion is required to keep the plugin calls `is_elementwise=True`, which is what makes
    `over` / `group_by` invoke them once per partition rather than as an aggregation. It also guards correctness:
    the Rust plugins zip their inputs element-wise via `try_*_elementwise`, which truncates to the shortest input rather
    than broadcasting a length-1 literal. Mixed with a column-valued input, a scalar `pl.lit(value)` would collapse the
    result to length 1 and silently drop every row past the first. Some Polars versions broadcast the literal upstream
    and hide this; not all do, so the plugin contract must own row-alignment rather than rely on it.

    Arguments:
        value: A Python scalar of one of `scalar_types`, or an `IntoExprColumn`.
        name: Input name, used only to build the error message.
        scalar_label: Human description of the accepted scalar (e.g. `"a float"`), for the error.
        scalar_types: Python scalar type(s) accepted for expansion. `bool` is always rejected
            (it is an `int` subclass but never a valid numeric input, so `scalar_types=int` still excludes it).
        dtype: Dtype for the expanded scalar; `None` lets Polars infer it from `value`.
    """
    if isinstance(value, pl.Expr):
        return value
    if isinstance(value, str):
        return pl.col(value)
    if isinstance(value, pl.Series):
        return pl.lit(value)
    if isinstance(value, scalar_types) and not isinstance(value, bool):
        return pl.repeat(value, n=pl.len(), dtype=dtype)
    msg = f"{name} should be {scalar_label} or IntoExprColumn (pl.Expr, str, pl.Series), found {type(value)}"
    raise TypeError(msg)


def coerce_param(value: float | IntoExprColumn, *, name: str) -> pl.Expr:
    """Coerce a float distribution parameter (e.g. `mu`, `sigma`, `p`) into a row-aligned `pl.Expr`.

    Accepts a strict Python `float` or an `IntoExprColumn`; an `int` or `bool` raises `TypeError`
    (a probability or scale is a float, not a count). See `_coerce` for the row-alignment rationale.
    """
    return _coerce(value, name=name, scalar_label="a float", scalar_types=float, dtype=pl.Float64())


def coerce_n(value: int | IntoExprColumn, *, name: str = "n") -> pl.Expr:
    """Coerce an integer count parameter (e.g. binomial trial count `n`) into a row-aligned `pl.Expr`.

    Accepts a Python `int` or an `IntoExprColumn`; a `bool` (an `int` subclass, but not a sensible count),
    a `float`, or any other type raises `TypeError`. The expanded scalar is `Int64`.
    The *value* (`>= 0`) is validated per row in Rust, not here. See `_coerce` for the rationale.
    """
    return _coerce(value, name=name, scalar_label="an int", scalar_types=int, dtype=pl.Int64())


def scalar_float(value: float | IntoExprColumn) -> float | None:
    """Return `value` as a `float` if it is a plain numeric scalar, else `None`.

    Used by samplers to detect a constant (non-`Expr`) parameter and route it through the
    constant-parameter fast path, which passes it as a plugin kwarg validated once in Rust rather
    than expanding it into a per-row `pl.repeat` column (see `_coerce`). `bool` is an `int` subclass
    but never a valid parameter, so it is excluded and falls back to the per-row path.
    """
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def scalar_int(value: int | IntoExprColumn) -> int | None:
    """Return `value` as an `int` if it is a plain integer scalar (excluding `bool`), else `None`.

    The integer-count counterpart to `scalar_float` (e.g. binomial `n`), for the same fast-path routing.
    """
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def scalar_kwargs(**params: float | None) -> dict[str, float | int] | None:
    """Bundle the constant-parameter fast-path kwargs: a dict when every parameter is a scalar, else `None`.

    Each keyword is a parameter already passed through `scalar_float` / `scalar_int`, so a `None`
    value marks a column-valued parameter. All-scalar parameters route `sample` through the
    `<name>_sample_scalar` plugin (validated once, passed as kwargs); a single column-valued one
    falls back to the general per-row plugin, so the whole bundle collapses to `None`.
    """
    return (
        None if any(value is None for value in params.values()) else {k: v for k, v in params.items() if v is not None}
    )


def as_expr(value: float | IntoExprColumn) -> pl.Expr:
    """Coerce a value-keyed method input (`value` / `quantile`) into a row-aligned `pl.Expr`.

    More permissive on scalars than `coerce_param`: a support point or quantile may be an `int` (`cdf(0)`, `pmf(1)`) or
    a `float`, so both are accepted and the expanded dtype is left to Polars to infer.

    A `bool` or other non-numeric type raises `TypeError`. See `_coerce` for the row-alignment rationale.
    """
    return _coerce(value, name="value", scalar_label="a number (int or float)", scalar_types=(int, float))


def register_plugin(
    function_name: str,
    args: IntoExprColumn | Iterable[IntoExprColumn],
    *,
    kwargs: Mapping[str, float | int | None] | None = None,
) -> pl.Expr:
    """Register a polars-stats Rust plugin call, fixing the defaults every distribution shares.

    Wraps `register_plugin_function` so each call site spells only what varies (the function name, its input exprs,
    and an optional sampler `seed` or constant-parameter kwargs). `plugin_path=LIB`, `is_elementwise` is fixed to
    `True`: every distribution plugin is per-row by contract (see `coerce_param`), and an aggregating plugin would
    break `over` / `group_by`, so this is a guard rather than a default.

    Arguments:
        function_name: The `#[polars_expr]` function exported by the Rust crate.
        args: One expr or an iterable of exprs forming the plugin's positional inputs.
        kwargs: Static keyword arguments serialised to Rust (a sampler `seed`, the multi-draw sampler's draw
            count `size`, and/or the constant parameters of a `*_scalar` fast-path plugin). Accepted as a
            `Mapping` so the narrower `_scalar_kwargs` dicts pass without an invariance fight;
            `register_plugin_function` wants a `dict`, hence the copy.
    """
    return register_plugin_function(
        plugin_path=LIB,
        function_name=function_name,
        args=args,
        kwargs=None if kwargs is None else dict(kwargs),
        is_elementwise=True,
    )


def propagate_null_and_nan(value: pl.Expr, result: pl.Expr) -> pl.Expr:
    """Return `result`, overridden to null/NaN where `value` is null/NaN respectively.

    Applied by every public value-keyed wrapper (scipy semantics: null in, null out; `NaN` in, `NaN` out).
    """
    guarded = pl.when(value.is_null()).then(pl.lit(None)).when(value.is_nan()).then(float("nan")).otherwise(result)
    name = value.meta.output_name(raise_if_undetermined=False)
    return guarded if name is None else guarded.alias(name)


class _UnivariateDistribution(ABC):
    """Abstract base class for a univariate probability distribution.

    Subclasses model a parameterised distribution and expose its standard functional forms as polars expressions.
    Parameters may be Python scalars or `pl.Expr`, which lets a single instance describe a different distribution per
    row (e.g. `Normal(mu=pl.col("mu"), sigma=1.0)`).

    The interface mirrors `scipy.stats.rv_continuous` / `rv_discrete` but returns `pl.Expr` instead of NumPy arrays.
    """

    _plugin_prefix: ClassVar[str]
    """Shared prefix of this distribution's Rust plugin names (typically snake-case distribution name, e.g. `"normal"`).

    Every plugin a distribution registers is expected to follow the pattern: `f"{_plugin_prefix}_{suffix}"`
    """

    _scalar_kwargs: dict[str, float | int] | None
    """Constant parameters for the `<prefix>_*_scalar` fast paths, `None` when any parameter is column-valued.

    Set by each subclass at construction via `scalar_kwargs`. Doubles as the routing switch shared by
    `sample`, `_samples`, and `_value_plugin` (all-scalar routes to the validated-once fast path), and
    as the naming switch: with no input column to inherit a name from, the sampler outputs take the
    deliberate default names `"sample"` / `"samples"`; column-valued parameters keep polars root-name
    semantics instead.
    """

    @property
    @abstractmethod
    def _param_exprs(self) -> tuple[pl.Expr, ...]:
        """The distribution's parameters as coerced, row-aligned exprs, in plugin-input order.

        The per-row plugins take these as positional inputs, ahead of `ROW_INDEX_EXPR` for the samplers
        and after `value` for the value-keyed methods.
        The order is part of the contract: output naming follows the first expression (polars root-name semantics,
        pinned by `output_name_test.py`), and the Rust side reads them positionally.
        Constant parameters additionally ride in `_scalar_kwargs` for the fast path.
        """

    def sample(self, seed: int | None = None) -> pl.Expr:
        """Draw one random variate per row.

        Returns a column with one variate per input row, in the distribution's element dtype
        (`Float64`, `UInt64` or `Boolean`). Output length follows the surrounding context (frame length
        under `select` / `with_columns`, partition length under `over` / `group_by`), and each row's draw
        is derived from a per-row sub-seed mixed from `seed` and the row's position, so the result is
        independent of Polars chunking and thread scheduling.

        A row with an invalid parameter raises; a row with a null parameter yields null. The output is
        named `"sample"` when every parameter is constant (the fast path); with any column-valued
        parameter the name follows the first parameter expression (polars root-name semantics, so
        `.name.*` modifiers keep working).
        """
        if self._scalar_kwargs is not None:
            return register_plugin(
                f"{self._plugin_prefix}_sample_scalar",
                (ROW_INDEX_EXPR,),
                kwargs={"seed": seed, **self._scalar_kwargs},
            ).alias("sample")
        return register_plugin(
            f"{self._plugin_prefix}_sample", (*self._param_exprs, ROW_INDEX_EXPR), kwargs={"seed": seed}
        )

    def samples(self, size: int, seed: int | None = None) -> pl.Expr:
        """Draw `size` random variates per row, returning `Array(inner=<element dtype>, shape=size)`.

        Each row's `size` draws are consecutive values from one per-row random stream keyed by `seed` and the
        row's position, so the result is reproducible for a fixed `seed` and independent of Polars chunking and
        thread scheduling. `samples(size=1)` matches `sample` for the same seed, and growing `size` extends each
        row's array without changing the existing draws.

        A row with a null parameter yields a null array (not an array of null elements), produced natively by
        the plugin via the output's outer validity; an invalid parameterisation raises.

        Naming follows `sample`: `"samples"` with all-constant parameters, the first parameter
        expression's root name otherwise.
        """
        if size <= 0:
            msg = f"size must be a positive integer, got {size}"
            raise ValueError(msg)
        out = self._samples(size=size, seed=seed)
        return out.alias("samples") if self._scalar_kwargs is not None else out

    def _samples(self, size: int, seed: int | None = None) -> pl.Expr:
        """Route the multi-draw call, returning a column of `Array(inner=..., shape=size)`.

        The draw semantics are documented on the public `samples`. Both parameter shapes run as one
        native plugin call returning the `Array` column directly: all-constant parameters route to
        `<name>_samples_scalar` (parameters validated once, in kwargs), column parameters to the
        per-row `<name>_samples` twin via `_samples_columns` (distribution rebuilt once per row, not
        once per draw). Seeding is positional on both paths, so their output is bit-identical
        (pinned by `test_samples_scalar_fast_path_matches_per_row`).
        """
        if self._scalar_kwargs is not None:
            return register_plugin(
                f"{self._plugin_prefix}_samples_scalar",
                (ROW_INDEX_EXPR,),
                kwargs={"seed": seed, "size": size, **self._scalar_kwargs},
            )
        return self._samples_columns(size=size, seed=seed)

    def _samples_columns(self, size: int, seed: int | None) -> pl.Expr:
        """Register the `<prefix>_samples` multi-draw plugin call for column-valued parameters.

        Mirrors `sample`'s per-row plugin shape: the parameter exprs plus `ROW_INDEX_EXPR` as inputs, the root
        `seed` and draw count `size` as kwargs. Called by `_samples`; naming is applied by `samples`, and a
        null-parameter row becomes a null array element inside the plugin.
        """
        return register_plugin(
            f"{self._plugin_prefix}_samples",
            (*self._param_exprs, ROW_INDEX_EXPR),
            kwargs={"seed": seed, "size": size},
        )

    def _value_plugin(self, function_name: str, value: pl.Expr) -> pl.Expr:
        """Register a value-keyed Rust plugin call `f(value, *params)` for a statrs-backed method.

        Parameters are validated inside the plugin, so every value-keyed method reports an invalid
        parameterisation consistently and propagates input nulls per row. With all-constant parameters the
        call routes to the `f"{function_name}_scalar"` twin (validated once, passed as kwargs, only `value`
        crosses FFI), bit-identical to the per-row path. Closed-form distributions (`Uniform`, `Bernoulli`)
        never call this; their hooks compute the formula directly in Polars.
        """
        return (
            register_plugin(f"{function_name}_scalar", (value,), kwargs=self._scalar_kwargs)
            if self._scalar_kwargs is not None
            else register_plugin(function_name, (value, *self._param_exprs))
        )

    def _scalar_lit_args(self) -> list[pl.Expr]:
        """The constant parameters as length-1 `pl.lit` exprs, in `_param_exprs` order.

        Only meaningful when every parameter is constant (`_scalar_kwargs is not None`); the callers
        guard on that. Passing length-1 literals to a validating or computing plugin makes its
        elementwise closure run **once**, not once per frame row (see `_checked`).

        Every subclass builds `_scalar_kwargs` in `_param_exprs` order, the same order the per-row
        plugin reads its inputs positionally, so the once-call and the per-row call validate the
        identical parameterisation.
        """
        assert self._scalar_kwargs is not None  # noqa: S101  # guarded by every caller
        return [pl.lit(value) for value in self._scalar_kwargs.values()]

    def _checked(self, plugin_name: str, validated: pl.Expr) -> pl.Expr:
        """A parameter-validating plugin call: per-row for column params, validated **once** for scalars.

        The validating plugins (`normal_sigma`, `uniform_range`, `bernoulli_proba`, ...) are
        elementwise and return the quantity they are named for (the validated `sigma`, the width
        `max - min`, the validated `p`, ...). Both paths are byte-identical for the same parameters:

        * **Column parameters**: the per-row plugin over `_param_exprs`, validating each row and
          propagating per-row nulls.
        * **All-scalar parameters**: the same plugin on length-1 `pl.lit` inputs, so it validates a
          single time and still raises the same `ComputeError` on an invalid constant. `validated`
          (a length-n expr recomputing that quantity from the raw parameter columns, e.g.
          `self._sigma` or `self._max - self._min`) is returned behind the length-1 validity gate;
          `pl.when` broadcasts the condition, so the result stays length-n and equals the per-row
          output element for element.
        """
        if self._scalar_kwargs is None:
            return register_plugin(plugin_name, self._param_exprs)
        validated_once = register_plugin(plugin_name, self._scalar_lit_args())
        return pl.when(validated_once.is_not_null()).then(validated)

    def cdf(self, value: float | IntoExprColumn) -> pl.Expr:
        """Cumulative distribution function, `P(X <= value)`. Nulls and NaNs in `value` are propagated."""
        v = as_expr(value)
        return propagate_null_and_nan(v, self._cdf(v))

    @abstractmethod
    def _cdf(self, value: pl.Expr) -> pl.Expr:
        """Core cdf formula on a coerced expr; null handling is applied by `cdf`."""

    def log_cdf(self, value: float | IntoExprColumn) -> pl.Expr:
        """Natural logarithm of the cdf. Nulls and NaNs in `value` are propagated."""
        v = as_expr(value)
        return propagate_null_and_nan(v, self._log_cdf(v))

    def _log_cdf(self, value: pl.Expr) -> pl.Expr:
        """Default `log(cdf)`; underflows to `-inf` once `cdf` rounds to `0` deep in the left tail.

        There is no native shortcut to bind (statrs exposes no `ln_cdf`, Polars no `erf`), so a
        distribution with a genuine tail must override this with a stable form: a `log1p` / exact
        closed form (Exponential, Uniform) or a special-function binding in Rust (Normal).
        """
        return self._cdf(value).log()

    def sf(self, value: float | IntoExprColumn) -> pl.Expr:
        """Survival function, `P(X > value) = 1 - cdf(value)`. Nulls and NaNs in `value` are propagated."""
        v = as_expr(value)
        return propagate_null_and_nan(v, self._sf(v))

    def _sf(self, value: pl.Expr) -> pl.Expr:
        """Default `1 - cdf`; subclasses override when a closed form is more accurate in the upper tail."""
        return 1 - self._cdf(value)

    def log_sf(self, value: float | IntoExprColumn) -> pl.Expr:
        """Natural logarithm of the survival function. Nulls and NaNs in `value` are propagated."""
        v = as_expr(value)
        return propagate_null_and_nan(v, self._log_sf(v))

    def _log_sf(self, value: pl.Expr) -> pl.Expr:
        """Default `log(sf)`; underflows to `-inf` once `sf` rounds to `0` deep in the right tail.

        Override it for any distribution with a genuine tail, where `log_sf` of a many-sigma event
        is the point. Same options as `_log_cdf`: a `log1p` / exact closed form or a Rust
        special-function binding.
        """
        return self._sf(value).log()

    def ppf(self, quantile: float | IntoExprColumn) -> pl.Expr:
        """Percent point function (inverse cdf).

        A `quantile` outside `[0, 1]` yields **null**. That is a guarantee, not an accident of the
        implementations: it holds for every distribution and both parameter regimes, and is pinned
        by `tests/property/ppf_domain_test.py`. Null is the right answer rather than an error
        because a quantile column is data, so one bad row must not fail the whole frame, and rather
        than `NaN` because `NaN` is reserved for a `NaN` *input* (below).

        The closed endpoints are in range and never null: `ppf(0)` and `ppf(1)` return the
        distribution's support bounds, which may be infinite (`Normal.ppf(0) = -inf`).

        Nulls are propagated and a `NaN` quantile yields `NaN`, matching scipy.
        """
        q = as_expr(quantile)
        return propagate_null_and_nan(q, self._ppf(q))

    @abstractmethod
    def _ppf(self, quantile: pl.Expr) -> pl.Expr:
        """Core inverse-cdf formula on a coerced expr; null handling is applied by `ppf`."""

    def isf(self, quantile: float | IntoExprColumn) -> pl.Expr:
        """Inverse survival function, the value `x` with `sf(x) == quantile`.

        Same domain contract as `ppf`, with the endpoints reversed: `quantile` outside `[0, 1]` yields
        null, nulls propagate, `NaN` yields `NaN`.
        """
        q = as_expr(quantile)
        return propagate_null_and_nan(q, self._isf(q))

    def _isf(self, quantile: pl.Expr) -> pl.Expr:
        """Fallback `ppf(1 - quantile)`, correct only where `quantile` is not tiny.

        `1 - quantile` resolves to `1.1e-16` absolute, so the tail mass a subclass is asked to invert
        is quantised to `1.1e-16 / quantile` relative before the inverse ever runs. Like `_cdf().log()`
        for `_log_cdf`, this is convenience rather than an implementation: inheriting it is a decision
        to justify. The forms that worked here, in rough order of preference, are a closed form
        (`Uniform`, `Bernoulli`, `Exponential`), a symmetry (`Normal`, and `LogNormal` by composing
        it), and entering an existing two-sided solve from the other tail.

        Being integer-valued or piecewise-linear does *not* make a distribution safe here, which was
        the tempting wrong conclusion: `Bernoulli(1e-17).isf(1e-20)` answered `0.0` where the answer
        is `1.0`, and `Uniform(-1, 0).isf(1e-17)` answered `0.0` against a true `-1e-17`. Prove the
        override unnecessary against `make audit`, with the parameter regime that would expose it,
        before skipping it. See docs/contributing.md, "Numerical stability".
        """
        return self._ppf(1 - quantile)

    @property
    def _checked_params(self) -> pl.Expr:
        """Single validating round-trip backing the closed-form moments (statrs-backed distributions only).

        A Rust plugin expr returning a parameter, or a derived quantity, that raises on an invalid
        parameterisation and is null when any parameter is null. `_moment` gates on it. A distribution whose
        moment formulas may omit a parameter (`Normal`, `LogNormal`, `Binomial`) overrides this and routes
        its moments through `_moment`; one whose validating expr is already part of every moment formula
        (`Uniform.range`, `Bernoulli._checked_p`) never calls `_moment`, so this default is never reached.
        """
        raise NotImplementedError

    def _moment(self, value: pl.Expr) -> pl.Expr:
        """Gate a closed-form moment on a non-null, valid parameterisation.

        Evaluating `_checked_params` validates the parameters in Rust (raising on an invalid
        parameterisation) and is null when any parameter is null, so the moment nulls on any null input and
        raises consistently with the value-keyed methods regardless of which parameters `value` itself
        references. Validation lives in the gate, so `value` reads the raw parameter exprs without
        re-validating.

        Gating on `_checked_params` alone suffices: every validator returns non-null only when *all*
        parameters are non-null (its `(Some, ..)` match arm), so an explicit per-parameter null check would
        be redundant (pinned by the `*_propagates_null_params` moment tests).
        """
        return pl.when(self._checked_params.is_not_null()).then(value)

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
        return self.ppf(0.5)

    @abstractmethod
    def entropy(self) -> pl.Expr:
        """Differential or Shannon entropy, in nats."""


class DiscreteDistribution(_UnivariateDistribution, ABC):
    """Abstract base class for discrete univariate distributions."""

    def pmf(self, value: float | IntoExprColumn) -> pl.Expr:
        """Probability mass function, `P(X = value)`. Nulls and NaNs in `value` are propagated."""
        v = as_expr(value)
        return propagate_null_and_nan(v, self._pmf(v))

    @abstractmethod
    def _pmf(self, value: pl.Expr) -> pl.Expr:
        """Core pmf formula on a coerced expr; null handling is applied by `pmf`."""

    def log_pmf(self, value: float | IntoExprColumn) -> pl.Expr:
        """Natural logarithm of the pmf. Nulls and NaNs in `value` are propagated."""
        v = as_expr(value)
        return propagate_null_and_nan(v, self._log_pmf(v))

    def _log_pmf(self, value: pl.Expr) -> pl.Expr:
        return self._pmf(value).log()


class ContinuousDistribution(_UnivariateDistribution, ABC):
    """Abstract base class for continuous univariate distributions."""

    def pdf(self, value: float | IntoExprColumn) -> pl.Expr:
        """Probability density function evaluated at `value`. Nulls and NaNs in `value` are propagated."""
        v = as_expr(value)
        return propagate_null_and_nan(v, self._pdf(v))

    @abstractmethod
    def _pdf(self, value: pl.Expr) -> pl.Expr:
        """Core pdf formula on a coerced expr; null handling is applied by `pdf`."""

    def log_pdf(self, value: float | IntoExprColumn) -> pl.Expr:
        """Natural logarithm of the pdf. Nulls and NaNs in `value` are propagated."""
        v = as_expr(value)
        return propagate_null_and_nan(v, self._log_pdf(v))

    def _log_pdf(self, value: pl.Expr) -> pl.Expr:
        return self._pdf(value).log()
