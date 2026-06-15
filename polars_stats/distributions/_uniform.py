from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import polars as pl

from polars_stats.distributions._base import (
    ContinuousDistribution,
    coerce_param,
    register_plugin,
    scalar_float,
    scalar_kwargs,
)

if TYPE_CHECKING:
    from polars_stats._typing import IntoExprColumn


class Uniform(ContinuousDistribution):
    """Continuous uniform distribution over ``[min, max]``.

    Equivalent to ``scipy.stats.uniform(loc=min, scale=max - min)``.

    Following scipy, the density, cdf and the other closed forms treat the support as the closed interval ``[min, max]``
    (so ``pdf(max) == 1 / (max - min)``); the ``sample`` plugin draws on the half-open ``[min, max)``.

    Arguments:
        min: Lower bound. Either a Python ``float`` or an ``IntoExprColumn`` (``pl.Expr``,
            ``pl.Series`` or column name ``str``) carrying one bound per row.
        max: Upper bound, with ``max > min``. Same accepted types as ``min``.

    An invalid parameterisation (``max <= min``, a non-finite bound, or a width ``max - min``
    overflowing ``float64``) is not checked at construction; matching every other distribution, it
    raises ``InvalidOperation`` (a ``ComputeError``) when any method is evaluated. Null bounds
    propagate to null.
    """

    _min: pl.Expr
    _max: pl.Expr
    _plugin_prefix: ClassVar[str] = "uniform"

    def __init__(self, min: float | IntoExprColumn, max: float | IntoExprColumn) -> None:  # noqa: A002
        self._min = coerce_param(min, name="min")
        self._max = coerce_param(max, name="max")
        # Constant bounds enable the constant-bounds sampler fast path; `None` falls back to the
        # per-row plugin.
        self._scalar_kwargs = scalar_kwargs(min=scalar_float(min), max=scalar_float(max))

    @property
    def _param_exprs(self) -> tuple[pl.Expr, ...]:
        return (self._min, self._max)

    @property
    def range(self) -> pl.Expr:
        """Width of the support, ``max - min``.

        Computed in Rust so an invalid parameterisation (``max <= min``, a non-finite bound, or a
        width overflowing ``float64``) raises rather than silently yielding a non-positive or
        infinite width. Every closed-form method derives from this, so they all validate
        consistently. Null bounds propagate.
        """
        return register_plugin("uniform_range", self._param_exprs)

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
