from __future__ import annotations

import random
from abc import ABC, abstractmethod
from itertools import repeat
from typing import TYPE_CHECKING, ClassVar

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterable

    from polars_stats._typing import IntoExprColumn, PolarsDataType

# TODO(FBruzzesi): Investigate better implementations for log_* methods over
# the naive ones due to concerns on numerical stability

_ROW_INDEX = "__polars_stats_row_index__"


def coerce_param(value: float | IntoExprColumn, *, name: str) -> pl.Expr:
    """Coerce a distribution parameter into a row-aligned `pl.Expr`.

    A Python `float` is expanded to a length-N expression (not `pl.lit`) so the plugin always
    receives a row-aligned input; this keeps plugin calls `is_elementwise=True`, which is what
    makes `over` / `group_by` invoke them once per partition rather than as an aggregation.

    Arguments:
        value: Either a Python `float` or an `IntoExprColumn` (`pl.Expr`, `pl.Series` or column
            name `str`).
        name: Parameter name, used only to build the error message.
    """
    if isinstance(value, float):
        return pl.repeat(value, n=pl.len(), dtype=pl.Float64())
    if isinstance(value, pl.Expr):
        return value
    if isinstance(value, pl.Series):
        return pl.lit(value)
    if isinstance(value, str):
        return pl.col(value)
    msg = f"{name} should be a float or IntoExprColumn (pl.Expr, str, pl.Series), found {type(value)}"
    raise TypeError(msg)


def as_expr(value: float | pl.Expr) -> pl.Expr:
    """Wrap a scalar method input as a literal; pass an existing `pl.Expr` through.

    Used for method arguments such as `value` / `quantile`, which broadcast against the row-aligned
    parameter expressions and so do not need the `pl.repeat` expansion that `coerce_param` applies.
    """
    return value if isinstance(value, pl.Expr) else pl.lit(value)


def propagate_null(value: pl.Expr, result: pl.Expr) -> pl.Expr:
    """Return `result`, overridden to null on rows where `value` is null.

    A null predicate in `pl.when` collapses to the `otherwise` branch, so a method whose
    `otherwise` is a non-null constant (e.g. `pdf` returning `0.0` outside the support) would
    silently emit that constant for a null input. Guarding on `value.is_null()` first makes every
    value-keyed method propagate input nulls uniformly.

    The null literal is untyped so the output dtype follows `result` (e.g. `Boolean` for a discrete
    `ppf`, `Float64` for a density).
    """
    return pl.when(value.is_null()).then(pl.lit(None)).otherwise(result)


def row_index_expr() -> pl.Expr:
    """Per-row position `0..len` used to derive per-row sub-seeds in samplers.

    Evaluated inside the surrounding context, so `pl.len()` is the frame length under
    `select` / `with_columns` and the partition length under `over` / `group_by`.
    """
    return pl.int_range(0, pl.len(), dtype=pl.UInt64).alias(_ROW_INDEX)


class _UnivariateDistribution(ABC):
    """Abstract base class for a univariate probability distribution.

    Subclasses model a parameterised distribution and expose its standard functional forms as polars expressions.
    Parameters may be Python scalars or `pl.Expr`, which lets a single instance describe a different distribution per
    row (e.g. `Normal(mu=pl.col("mu"), sigma=1.0)`).

    The interface mirrors `scipy.stats.rv_continuous` / `rv_discrete` but returns `pl.Expr` instead of NumPy arrays.
    """

    _sample_dtype: ClassVar[PolarsDataType]
    """Element dtype produced `sample` (e.g. `Boolean`, `Float64`, `UInt64`). Set by each subclass."""

    @abstractmethod
    def _valid_mask(self) -> pl.Expr:
        """Boolean expr, `True` on rows whose parameters yield a well-defined draw.

        Rows that are `False` (null or out-of-domain parameters) get a null array from `samples`.
        """

    @abstractmethod
    def sample(self, seed: int | None = None) -> pl.Expr:
        """Draw one random variate per row.

        Returns a polars Expr evaluating to a column with one variate per input row.
        """

    def samples(self, size: int, seed: int | None = None) -> pl.Expr:
        """Draw `size` random variates per row, returning `Array(inner=_sample_dtype, shape=size)`.

        When `seed` is set, distinct sub-seeds are derived from it so the `size` underlying `sample`
        calls produce independent streams. Without this, every plugin call would re-seed the same RNG
        and yield `size` identical columns.

        A row whose parameters are invalid (see `_valid_mask`) yields a null array, not an array of
        null elements.
        """
        if size <= 0:
            msg = f"size must be a positive integer, got {size}"
            raise ValueError(msg)
        return (
            pl.when(self._valid_mask())
            .then(self._samples(size=size, seed=seed))
            .otherwise(pl.lit(None, dtype=pl.Array(self._sample_dtype, shape=size)))
        )

    def _samples(self, size: int, seed: int | None = None) -> pl.Expr:
        """Draw `size` random variates per row.

        Returns polars Expr evaluating to a column of `Array(inner=..., shape=size)`.

        When ``seed`` is set, distinct sub-seeds are derived from it so the `size` underlying ``sample`` calls
        produce independent streams. Without this, every plugin call would re-seed the same RNG and yield
        ``size`` identical columns.
        """
        rng = random.Random(seed)  # noqa: S311
        seeds: Iterable[int] | Iterable[None] = (
            repeat(None, size) if seed is None else (rng.randrange(2**63) for _ in range(size))
        )

        return pl.concat_arr(self.sample(seed=s) for s in seeds)

    def cdf(self, value: float | pl.Expr) -> pl.Expr:
        """Cumulative distribution function, `P(X <= value)`. Nulls in `value` are propagated."""
        v = as_expr(value)
        return propagate_null(v, self._cdf(v))

    @abstractmethod
    def _cdf(self, value: pl.Expr) -> pl.Expr:
        """Core cdf formula on a coerced expr; null handling is applied by `cdf`."""

    def log_cdf(self, value: float | pl.Expr) -> pl.Expr:
        """Natural logarithm of the cdf. Nulls in `value` are propagated."""
        v = as_expr(value)
        return propagate_null(v, self._log_cdf(v))

    def _log_cdf(self, value: pl.Expr) -> pl.Expr:
        return self._cdf(value).log()

    def sf(self, value: float | pl.Expr) -> pl.Expr:
        """Survival function, `P(X > value) = 1 - cdf(value)`. Nulls in `value` are propagated."""
        v = as_expr(value)
        return propagate_null(v, self._sf(v))

    def _sf(self, value: pl.Expr) -> pl.Expr:
        """Default `1 - cdf`; subclasses override when a closed form is more accurate in the upper tail."""
        return 1 - self._cdf(value)

    def log_sf(self, value: float | pl.Expr) -> pl.Expr:
        """Natural logarithm of the survival function. Nulls in `value` are propagated."""
        v = as_expr(value)
        return propagate_null(v, self._log_sf(v))

    def _log_sf(self, value: pl.Expr) -> pl.Expr:
        return self._sf(value).log()

    def ppf(self, quantile: float | pl.Expr) -> pl.Expr:
        """Percent point function (inverse cdf).

        `quantile` is expected to lie in `[0, 1]`; nulls are propagated. Behaviour for out-of-range
        quantiles is implementation-defined and should not be relied on; callers are responsible for
        bounding `quantile` upstream when the source allows invalid values.
        """
        q = as_expr(quantile)
        return propagate_null(q, self._ppf(q))

    @abstractmethod
    def _ppf(self, quantile: pl.Expr) -> pl.Expr:
        """Core inverse-cdf formula on a coerced expr; null handling is applied by `ppf`."""

    def isf(self, quantile: float | pl.Expr) -> pl.Expr:
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

    def pmf(self, value: float | pl.Expr) -> pl.Expr:
        """Probability mass function, `P(X = value)`. Nulls in `value` are propagated."""
        v = as_expr(value)
        return propagate_null(v, self._pmf(v))

    @abstractmethod
    def _pmf(self, value: pl.Expr) -> pl.Expr:
        """Core pmf formula on a coerced expr; null handling is applied by `pmf`."""

    def log_pmf(self, value: float | pl.Expr) -> pl.Expr:
        """Natural logarithm of the pmf. Nulls in `value` are propagated."""
        v = as_expr(value)
        return propagate_null(v, self._log_pmf(v))

    def _log_pmf(self, value: pl.Expr) -> pl.Expr:
        return self._pmf(value).log()


class ContinuousDistribution(_UnivariateDistribution, ABC):
    """Abstract base class for continuous univariate distributions."""

    def pdf(self, value: float | pl.Expr) -> pl.Expr:
        """Probability density function evaluated at `value`. Nulls in `value` are propagated."""
        v = as_expr(value)
        return propagate_null(v, self._pdf(v))

    @abstractmethod
    def _pdf(self, value: pl.Expr) -> pl.Expr:
        """Core pdf formula on a coerced expr; null handling is applied by `pdf`."""

    def log_pdf(self, value: float | pl.Expr) -> pl.Expr:
        """Natural logarithm of the pdf. Nulls in `value` are propagated."""
        v = as_expr(value)
        return propagate_null(v, self._log_pdf(v))

    def _log_pdf(self, value: pl.Expr) -> pl.Expr:
        return self._pdf(value).log()
