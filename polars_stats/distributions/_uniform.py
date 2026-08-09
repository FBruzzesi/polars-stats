from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar

import polars as pl

from polars_stats.distributions._base import (
    ContinuousDistribution,
    coerce_param,
    scalar_float,
    scalar_kwargs,
)

if TYPE_CHECKING:
    from polars_stats._typing import IntoExprColumn

_MEDIAN_QUANTILE = 0.5
"""Where `Uniform._inverse` switches which bound it interpolates from; see that method."""


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
        self._scalar_kwargs = scalar_kwargs(min=scalar_float(min), max=scalar_float(max))

    @property
    def _param_exprs(self) -> tuple[pl.Expr, ...]:
        return (self._min, self._max)

    @property
    def range(self) -> pl.Expr:
        """Width of the support, ``max - min``.

        Validated in Rust so an invalid parameterisation (``max <= min``, a non-finite bound, or a
        width overflowing ``float64``) raises rather than silently yielding a non-positive or
        infinite width. Every closed-form method (moments and pdf/cdf/ppf) derives from this, so they
        all validate consistently; null bounds propagate. See ``_checked`` for the scalar-vs-column
        routing.
        """
        return self._checked("uniform_range", self._max - self._min)

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

    def _log_cdf(self, value: pl.Expr) -> pl.Expr:
        """Log of the cdf ratio, through ``log1p`` of the survival ratio once past the midpoint.

        ``-inf`` at/below ``min``, ``0`` at/above ``max``. Overrides the base ``cdf().log()``, which
        loses the whole near-certain half: approaching ``max`` the ratio rounds to exactly ``1`` and
        its log to ``0``, where the truth is a small negative. Same branch ``normal.rs``'s
        ``ln_half_erfc`` takes.
        """
        upper = (-((self._max - value) / self.range)).log1p()
        lower = ((value - self._min) / self.range).log()
        return (
            pl.when(value < self._min)
            .then(float("-inf"))
            .when(value >= self._max)
            .then(0.0)
            .when(value > self._min + self.range / 2)
            .then(upper)
            .otherwise(lower)
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
        """``0`` at/below ``min``, ``-inf`` at/above ``max``, and the mirror of ``_log_cdf`` between.

        The near-certain half is the *lower* one here, so the ``log1p`` branch is on ``value`` below
        the midpoint; the plain log of the ratio is exact on the other side.
        """
        upper = ((self._max - value) / self.range).log()
        lower = (-((value - self._min) / self.range)).log1p()
        return (
            pl.when(value < self._min)
            .then(0.0)
            .when(value >= self._max)
            .then(float("-inf"))
            .when(value > self._min + self.range / 2)
            .then(upper)
            .otherwise(lower)
        )

    def _inverse(self, quantile: pl.Expr, *, ascending: bool) -> pl.Expr:
        """Interpolate from whichever bound the answer is nearest; null outside ``[0, 1]``.

        Both inverses are one multiply-add, and both lose the answer when they interpolate from the
        *far* bound: ``min + quantile * range`` is a difference of nearly equal numbers once the
        result lands near ``max``. Anchoring to the near bound makes the addition well conditioned,
        and the ``1 - quantile`` it needs is only formed above the median, where Sterbenz makes it
        exact.

        ``ascending`` is ``True`` for ``ppf`` (small quantiles sit at ``min``) and ``False`` for
        ``isf`` (small quantiles sit at ``max``).
        """
        lo, hi = (self._min, self._max) if ascending else (self._max, self._min)
        step = self.range if ascending else -self.range
        return (
            pl.when(quantile.is_between(0, 1))
            .then(
                pl.when(quantile <= _MEDIAN_QUANTILE).then(lo + quantile * step).otherwise(hi - (1 - quantile) * step)
            )
            .otherwise(pl.lit(None, dtype=pl.Float64()))
        )

    def _ppf(self, quantile: pl.Expr) -> pl.Expr:
        """``min + quantile * (max - min)``; null for ``quantile`` outside ``[0, 1]``.

        Evaluated as ``max - (1 - quantile) * range`` above the median, so a result near ``max`` on a
        wide span keeps its relative precision. See ``_inverse``.
        """
        return self._inverse(quantile, ascending=True)

    def _isf(self, quantile: pl.Expr) -> pl.Expr:
        """``max - quantile * (max - min)``; null for ``quantile`` outside ``[0, 1]``.

        The mirror of ``_ppf``, not the base-class ``ppf(1 - quantile)``: below
        ``quantile ~ 1.1e-16`` that complement rounds to exactly ``1.0`` and the whole tail
        collapses onto ``min + range``, which is total wherever ``max`` is near zero
        (``Uniform(-1, 0).isf(1e-17)`` returned ``0.0`` against a true ``-1e-17``). See ``_inverse``.
        """
        return self._inverse(quantile, ascending=False)

    def mean(self) -> pl.Expr:
        """Expected value, ``(min + max) / 2``."""
        return self._min + self.range / 2

    def variance(self) -> pl.Expr:
        """Variance, ``(max - min)^2 / 12``."""
        return self.range**2 / 12

    def std(self) -> pl.Expr:
        """Standard deviation, ``(max - min) / sqrt(12)``.

        Overrides the base-class ``variance().sqrt()``, which squares the span and then unsquares
        it: the round trip saturates about 300 decades before the answer does. Dividing by
        ``sqrt(12)`` once also drops a rounding.
        """
        return self.range / math.sqrt(12)

    def median(self) -> pl.Expr:
        """Median, ``(min + max) / 2``."""
        return self._min + self.range / 2

    def entropy(self) -> pl.Expr:
        """Differential entropy, ``log(max - min)``."""
        return self.range.log()
