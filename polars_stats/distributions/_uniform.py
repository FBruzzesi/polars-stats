from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar

from polars_stats.distributions._base import (
    ContinuousDistribution,
    coerce_param,
    scalar_float,
    scalar_kwargs,
)

if TYPE_CHECKING:
    import polars as pl

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
    raises ``InvalidOperation`` (a ``ComputeError``) when any method is evaluated.

    A null bound propagates to null wherever the result depends on it. Where the *other*, known
    bound already places the evaluation point outside the support, the bound-free constant survives
    instead (``pdf`` ``0``, ``log_pdf`` ``-inf``, and the saturated end of ``cdf`` / ``log_cdf`` /
    ``sf`` / ``log_sf``). Both inverses null at every quantile under either null bound.
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
        infinite width. The moments derive from this; the value-keyed methods validate inside their own
        plugin, through the same Rust check. Null bounds propagate. See ``_checked`` for the
        scalar-vs-column routing.
        """
        return self._checked("uniform_range", self._max - self._min)

    def _pdf(self, value: pl.Expr) -> pl.Expr:
        """``1 / (max - min)`` on the closed support ``[min, max]``, ``0`` outside."""
        return self._value_plugin("uniform_pdf", value)

    def _log_pdf(self, value: pl.Expr) -> pl.Expr:
        """``-log(max - min)`` on the closed support, ``-inf`` outside."""
        return self._value_plugin("uniform_ln_pdf", value)

    def _cdf(self, value: pl.Expr) -> pl.Expr:
        """``(value - min) / (max - min)``, clamped to ``0`` below ``min`` and ``1`` from ``max`` up."""
        return self._value_plugin("uniform_cdf", value)

    def _log_cdf(self, value: pl.Expr) -> pl.Expr:
        """``log(cdf)`` below the midpoint, ``log1p(-sf)`` above it; ``-inf`` at/below ``min``, ``0`` from ``max`` up.

        Overrides the base ``cdf().log()``, which loses the whole near-certain half: approaching
        ``max`` the ratio rounds to exactly ``1`` and its log to ``0``, where the truth is a small
        negative.
        """
        return self._value_plugin("uniform_ln_cdf", value)

    def _sf(self, value: pl.Expr) -> pl.Expr:
        """``(max - value) / (max - min)``, clamped to ``1`` below ``min`` and ``0`` from ``max`` up."""
        return self._value_plugin("uniform_sf", value)

    def _log_sf(self, value: pl.Expr) -> pl.Expr:
        """``0`` at/below ``min``, ``-inf`` from ``max`` up, and the mirror of ``_log_cdf`` between."""
        return self._value_plugin("uniform_ln_sf", value)

    def _ppf(self, quantile: pl.Expr) -> pl.Expr:
        """``min + quantile * (max - min)``; null for ``quantile`` outside ``[0, 1]``.

        Evaluated as ``max - (1 - quantile) * range`` above the median, so a result near ``max`` on a
        wide span keeps its relative precision.
        """
        return self._value_plugin("uniform_ppf", quantile)

    def _isf(self, quantile: pl.Expr) -> pl.Expr:
        """``max - quantile * (max - min)``; null for ``quantile`` outside ``[0, 1]``.

        The mirror of ``_ppf``, not the base-class ``ppf(1 - quantile)``: below
        ``quantile ~ 1.1e-16`` that complement rounds to exactly ``1.0`` and the whole tail
        collapses onto ``min + range``, which is total wherever ``max`` is near zero
        (``Uniform(-1, 0).isf(1e-17)`` returned ``0.0`` against a true ``-1e-17``).
        """
        return self._value_plugin("uniform_isf", quantile)

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
