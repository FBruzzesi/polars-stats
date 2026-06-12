from __future__ import annotations

import random
from abc import ABC, abstractmethod
from itertools import repeat
from typing import TYPE_CHECKING, ClassVar

import polars as pl
from polars.plugins import register_plugin_function

from polars_stats._lib import LIB

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from polars_stats._typing import IntoExprColumn, PolarsDataType

# TODO(FBruzzesi): Investigate better implementations for log_* methods over
# the naive ones due to concerns on numerical stability

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
    """Coerce a float distribution parameter (e.g. `mean`, `sigma`, `p`) into a row-aligned `pl.Expr`.

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
    kwargs: Mapping[str, float | int | list[int | None] | None] | None = None,
) -> pl.Expr:
    """Register a polars-stats Rust plugin call, fixing the defaults every distribution shares.

    Wraps `register_plugin_function` so each call site spells only what varies (the function name, its input exprs,
    and an optional sampler `seed` or constant-parameter kwargs). `plugin_path=LIB`, `is_elementwise` is fixed to
    `True`: every distribution plugin is per-row by contract (see `coerce_param`), and an aggregating plugin would
    break `over` / `group_by`, so this is a guard rather than a default.

    Arguments:
        function_name: The `#[polars_expr]` function exported by the Rust crate.
        args: One expr or an iterable of exprs forming the plugin's positional inputs.
        kwargs: Static keyword arguments serialised to Rust (a sampler `seed`, the multi-draw sampler's `seeds`
            list, and/or the constant parameters of a `*_scalar` fast-path plugin). Accepted as a `Mapping` so the
            narrower `_scalar_kwargs` dicts pass without an invariance fight; `register_plugin_function` wants a
            `dict`, hence the copy.
    """
    return register_plugin_function(
        plugin_path=LIB,
        function_name=function_name,
        args=args,
        kwargs=None if kwargs is None else dict(kwargs),
        is_elementwise=True,
    )


def propagate_null(value: pl.Expr, result: pl.Expr) -> pl.Expr:
    """Return `result`, overridden to null on rows where `value` is null.

    A null predicate in `pl.when` collapses to the `otherwise` branch, so a method whose `otherwise` is a non-null
    constant (e.g. `pdf` returning `0.0` outside the support) would silently emit that constant for a null input.

    Guarding on `value.is_null()` first makes every value-keyed method propagate input nulls uniformly.

    The null literal is untyped so the output dtype follows `result`.
    """
    return pl.when(value.is_null()).then(pl.lit(None)).otherwise(result)


class _UnivariateDistribution(ABC):
    """Abstract base class for a univariate probability distribution.

    Subclasses model a parameterised distribution and expose its standard functional forms as polars expressions.
    Parameters may be Python scalars or `pl.Expr`, which lets a single instance describe a different distribution per
    row (e.g. `Normal(mu=pl.col("mu"), sigma=1.0)`).

    The interface mirrors `scipy.stats.rv_continuous` / `rv_discrete` but returns `pl.Expr` instead of NumPy arrays.
    """

    _sample_dtype: ClassVar[PolarsDataType]
    """Element dtype produced `sample` (e.g. `Boolean`, `Float64`, `UInt64`). Set by each subclass."""

    _samples_scalar_plugin: ClassVar[str]
    """Name of the `<name>_samples_scalar` multi-draw plugin backing the constant-parameter `samples` fast path.

    Set by each subclass; `_samples` routes through it when `_scalar_kwargs` is non-`None`.
    """

    _scalar_kwargs: dict[str, float | int] | None
    """Constant parameters for the `<name>_sample_scalar` fast path, `None` when any parameter is column-valued.

    Set by each subclass at construction via `scalar_kwargs`. Doubles as the naming switch: with no
    input column to inherit a name from, the sampler outputs get the deliberate default names
    `"sample"` / `"samples"`; column-valued parameters keep polars root-name semantics instead.
    """

    @abstractmethod
    def _valid_mask(self) -> pl.Expr:
        """Boolean expr, `True` on rows whose parameters yield a well-defined draw.

        Rows that are `False` (null or out-of-domain parameters) get a null array from `samples`.
        """

    @abstractmethod
    def sample(self, seed: int | None = None) -> pl.Expr:
        """Draw one random variate per row.

        Returns a polars Expr evaluating to a column with one variate per input row. With
        all-constant parameters the column is named `"sample"`; with column-valued parameters the
        name follows the first parameter expression, as for any polars expression (so multi-column
        parameters and `.name.*` modifiers keep working).
        """

    def samples(self, size: int, seed: int | None = None) -> pl.Expr:
        """Draw `size` random variates per row, returning `Array(inner=_sample_dtype, shape=size)`.

        When `seed` is set, distinct sub-seeds are derived from it so the `size` underlying `sample`
        calls produce independent streams. Without this, every plugin call would re-seed the same RNG
        and yield `size` identical columns.

        A row whose parameters are invalid (see `_valid_mask`) yields a null array, not an array of null elements.

        Naming follows `sample`: `"samples"` with all-constant parameters, the first parameter
        expression's root name otherwise.
        """
        if size <= 0:
            msg = f"size must be a positive integer, got {size}"
            raise ValueError(msg)
        out = (
            pl.when(self._valid_mask())
            .then(self._samples(size=size, seed=seed))
            .otherwise(pl.lit(None, dtype=pl.Array(self._sample_dtype, shape=size)))
        )
        return out.alias("samples") if self._scalar_kwargs is not None else out

    def _samples(self, size: int, seed: int | None = None) -> pl.Expr:
        """Draw `size` random variates per row.

        Returns polars Expr evaluating to a column of `Array(inner=..., shape=size)`.

        When ``seed`` is set, distinct sub-seeds are derived from it so the ``size`` underlying draws produce
        independent streams. Without this, every draw would re-seed the same RNG and yield ``size`` identical
        columns. With ``seed=None``, each sub-seed resolves to fresh OS entropy (``size`` independent root seeds).

        All-constant parameters route to the ``<name>_samples_scalar`` multi-draw plugin: one plugin call returning
        the ``Array`` column directly, instead of ``size`` `sample` plugin calls glued by ``concat_arr`` (which pay
        the fixed expression/FFI cost ``size`` times). Draw ``j`` of row ``i`` is seeded ``(seed_j, i)`` on both
        paths, so their output is bit-identical (pinned by ``test_samples_scalar_fast_path_matches_per_row``).
        """
        rng = random.Random(seed)  # noqa: S311
        seeds: list[int | None] = (
            list(repeat(None, size)) if seed is None else [rng.randrange(2**63) for _ in range(size)]
        )

        if self._scalar_kwargs is not None:
            return register_plugin(
                self._samples_scalar_plugin, (ROW_INDEX_EXPR,), kwargs={"seeds": seeds, **self._scalar_kwargs}
            )
        return pl.concat_arr(self.sample(seed=s) for s in seeds)

    def cdf(self, value: float | IntoExprColumn) -> pl.Expr:
        """Cumulative distribution function, `P(X <= value)`. Nulls in `value` are propagated."""
        v = as_expr(value)
        return propagate_null(v, self._cdf(v))

    @abstractmethod
    def _cdf(self, value: pl.Expr) -> pl.Expr:
        """Core cdf formula on a coerced expr; null handling is applied by `cdf`."""

    def log_cdf(self, value: float | IntoExprColumn) -> pl.Expr:
        """Natural logarithm of the cdf. Nulls in `value` are propagated."""
        v = as_expr(value)
        return propagate_null(v, self._log_cdf(v))

    def _log_cdf(self, value: pl.Expr) -> pl.Expr:
        return self._cdf(value).log()

    def sf(self, value: float | IntoExprColumn) -> pl.Expr:
        """Survival function, `P(X > value) = 1 - cdf(value)`. Nulls in `value` are propagated."""
        v = as_expr(value)
        return propagate_null(v, self._sf(v))

    def _sf(self, value: pl.Expr) -> pl.Expr:
        """Default `1 - cdf`; subclasses override when a closed form is more accurate in the upper tail."""
        return 1 - self._cdf(value)

    def log_sf(self, value: float | IntoExprColumn) -> pl.Expr:
        """Natural logarithm of the survival function. Nulls in `value` are propagated."""
        v = as_expr(value)
        return propagate_null(v, self._log_sf(v))

    def _log_sf(self, value: pl.Expr) -> pl.Expr:
        return self._sf(value).log()

    def ppf(self, quantile: float | IntoExprColumn) -> pl.Expr:
        """Percent point function (inverse cdf).

        `quantile` is expected to lie in `[0, 1]`; nulls are propagated. Behaviour for out-of-range quantiles is
        implementation-defined and should not be relied on; callers are responsible for bounding `quantile` upstream
        when the source allows invalid values.
        """
        q = as_expr(quantile)
        return propagate_null(q, self._ppf(q))

    @abstractmethod
    def _ppf(self, quantile: pl.Expr) -> pl.Expr:
        """Core inverse-cdf formula on a coerced expr; null handling is applied by `ppf`."""

    def isf(self, quantile: float | IntoExprColumn) -> pl.Expr:
        """Inverse survival function, `ppf(1 - quantile)`. Nulls in `quantile` are propagated."""
        q = as_expr(quantile)
        return propagate_null(q, self._isf(q))

    def _isf(self, quantile: pl.Expr) -> pl.Expr:
        return self._ppf(1 - quantile)

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
        """Probability mass function, `P(X = value)`. Nulls in `value` are propagated."""
        v = as_expr(value)
        return propagate_null(v, self._pmf(v))

    @abstractmethod
    def _pmf(self, value: pl.Expr) -> pl.Expr:
        """Core pmf formula on a coerced expr; null handling is applied by `pmf`."""

    def log_pmf(self, value: float | IntoExprColumn) -> pl.Expr:
        """Natural logarithm of the pmf. Nulls in `value` are propagated."""
        v = as_expr(value)
        return propagate_null(v, self._log_pmf(v))

    def _log_pmf(self, value: pl.Expr) -> pl.Expr:
        return self._pmf(value).log()


class ContinuousDistribution(_UnivariateDistribution, ABC):
    """Abstract base class for continuous univariate distributions."""

    def pdf(self, value: float | IntoExprColumn) -> pl.Expr:
        """Probability density function evaluated at `value`. Nulls in `value` are propagated."""
        v = as_expr(value)
        return propagate_null(v, self._pdf(v))

    @abstractmethod
    def _pdf(self, value: pl.Expr) -> pl.Expr:
        """Core pdf formula on a coerced expr; null handling is applied by `pdf`."""

    def log_pdf(self, value: float | IntoExprColumn) -> pl.Expr:
        """Natural logarithm of the pdf. Nulls in `value` are propagated."""
        v = as_expr(value)
        return propagate_null(v, self._log_pdf(v))

    def _log_pdf(self, value: pl.Expr) -> pl.Expr:
        return self._pdf(value).log()
