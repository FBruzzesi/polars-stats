from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import polars as pl

from polars_stats.distributions._base import (
    ROW_INDEX_EXPR,
    ContinuousDistribution,
    coerce_param,
    register_plugin,
    scalar_float,
)

if TYPE_CHECKING:
    from polars_stats._typing import IntoExprColumn, PolarsDataType


class Uniform(ContinuousDistribution):
    """Continuous uniform distribution over ``[min, max]``.

    Equivalent to ``scipy.stats.uniform(loc=min, scale=max - min)``.

    Following scipy, the density, cdf and the other closed forms treat the support as the closed interval ``[min, max]``
    (so ``pdf(max) == 1 / (max - min)``); the ``sample`` plugin draws on the half-open ``[min, max)``.

    Arguments:
        min: Lower bound. Either a Python ``float`` or an ``IntoExprColumn`` (``pl.Expr``,
            ``pl.Series`` or column name ``str``) carrying one bound per row.
        max: Upper bound, with ``max > min``. Same accepted types as ``min``.

    An invalid parameterisation (``max <= min`` or a non-finite bound) is not checked at construction;
    matching every other distribution, it raises ``InvalidOperation`` (a ``ComputeError``) when any
    method is evaluated. Null bounds propagate to null.
    """

    _min: pl.Expr
    _max: pl.Expr
    _sample_dtype: ClassVar[PolarsDataType] = pl.Float64()

    def __init__(self, min: float | IntoExprColumn, max: float | IntoExprColumn) -> None:  # noqa: A002
        self._min = coerce_param(min, name="min")
        self._max = coerce_param(max, name="max")
        # Raw scalar bounds (if any) enable the constant-bounds sampler fast path; `None` falls back
        # to the per-row plugin.
        self._min_scalar = scalar_float(min)
        self._max_scalar = scalar_float(max)

    @property
    def range(self) -> pl.Expr:
        """Width of the support, ``max - min``.

        Computed in Rust so an invalid parameterisation (``max <= min`` or a non-finite bound) raises
        rather than silently yielding a non-positive width. Every closed-form method derives from this,
        so they all validate consistently. Null bounds propagate.
        """
        return register_plugin("uniform_range", (self._min, self._max))

    def _valid_mask(self) -> pl.Expr:
        # Only null bounds are masked to a null array here; `max <= min` raises via `range`.
        return self._min.is_not_null() & self._max.is_not_null()

    def sample(self, seed: int | None = None) -> pl.Expr:
        """Draw one uniform sample per row, returning a ``Float64`` column.

        Output length follows the surrounding context (frame length under ``select`` / ``with_columns``,
        partition length under ``over`` / ``group_by``). Each row's draw is derived from a per-row sub-seed mixed from
        ``seed`` and the row's position, so the result is independent of Polars chunking and thread scheduling.
        """
        if self._min_scalar is not None and self._max_scalar is not None:
            # Constant bounds: pass them as kwargs and validate once in Rust, instead of expanding
            # each into a full-length `pl.repeat` column re-validated per row. Only the row index
            # crosses FFI, so seeding (and thus output) is identical to the per-row path.
            return register_plugin(
                "uniform_sample_scalar",
                (ROW_INDEX_EXPR,),
                kwargs={"seed": seed, "min": self._min_scalar, "max": self._max_scalar},
            )
        return register_plugin("uniform_sample", (self._min, self._max, ROW_INDEX_EXPR), kwargs={"seed": seed})

    def _pdf(self, value: pl.Expr) -> pl.Expr:
        """``1 / (max - min)`` on ``[min, max]``, ``0`` outside."""
        in_range = value.is_between(self._min, self._max, closed="both")
        return pl.when(in_range).then(1 / self.range).otherwise(0.0)

    def _log_pdf(self, value: pl.Expr) -> pl.Expr:
        """``-log(max - min)`` on the support, ``-inf`` outside."""
        in_range = value.is_between(self._min, self._max, closed="both")
        return pl.when(in_range).then(-self.range.log()).otherwise(float("-inf"))

    def _cdf(self, value: pl.Expr) -> pl.Expr:
        """``(value - min) / (max - min)`` clamped to ``[0, 1]``."""
        return (
            pl.when(value < self._min)
            .then(0.0)
            .when(value >= self._max)
            .then(1.0)
            .otherwise((value - self._min) / self.range)
        )

    def _sf(self, value: pl.Expr) -> pl.Expr:
        """``(max - value) / (max - min)`` clamped to ``[0, 1]`` (closed form, accurate in the upper tail)."""
        return (
            pl.when(value < self._min)
            .then(1.0)
            .when(value >= self._max)
            .then(0.0)
            .otherwise((self._max - value) / self.range)
        )

    def _log_sf(self, value: pl.Expr) -> pl.Expr:
        """``0`` at/below ``min``, ``-inf`` at/above ``max``, ``log((max - value) / (max - min))`` between."""
        return (
            pl.when(value < self._min)
            .then(0.0)
            .when(value >= self._max)
            .then(float("-inf"))
            .otherwise(((self._max - value) / self.range).log())
        )

    def _ppf(self, quantile: pl.Expr) -> pl.Expr:
        """``min + quantile * (max - min)``; null for ``quantile`` outside ``[0, 1]``."""
        return (
            pl.when(quantile.is_between(0, 1))
            .then(self._min + quantile * self.range)
            .otherwise(pl.lit(None, dtype=pl.Float64()))
        )

    def mean(self) -> pl.Expr:
        """Expected value, ``(min + max) / 2``."""
        return self._min + self.range / 2

    def variance(self) -> pl.Expr:
        """Variance, ``(max - min)^2 / 12``."""
        return self.range**2 / 12

    def median(self) -> pl.Expr:
        """Median, ``(min + max) / 2``."""
        return self._min + self.range / 2

    def entropy(self) -> pl.Expr:
        """Differential entropy, ``log(max - min)``."""
        return self.range.log()
