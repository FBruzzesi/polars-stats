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
    from polars_stats._typing import DistributionName, IntoExprColumn

# Multiplied by, never divided against: polars spells `column / 3` as a reciprocal multiply on some engine and
# row-count combinations, which would make a moment depend on the frame's height.
_ONE_THIRD = 1 / 3
_ONE_EIGHTEENTH = 1 / 18
_ROOT_ONE_EIGHTEENTH = math.sqrt(_ONE_EIGHTEENTH)
_HALF_MINUS_LN_2 = 0.5 - math.log(2.0)


class Triangular(ContinuousDistribution):
    """Triangular distribution on ``[min, max]`` with its peak at ``mode``.

    Equivalent to ``scipy.stats.triang(c=(mode - min) / (max - min), loc=min, scale=max - min)`` and to numpy's
    ``triangular(left=min, mode=mode, right=max)``.

    Arguments:
        min: Lower end of the support. Either a Python ``float`` or an ``IntoExprColumn`` (``pl.Expr``, ``pl.Series``
            or column name ``str``) carrying one value per row.
        mode: The peak, with ``min <= mode <= max``; either end is allowed. Same accepted types as ``min``.
        max: Upper end of the support, with ``max > min``. Same accepted types as ``min``.

    Where ``mode`` sits on an end, the density there is its one-sided limit ``2 / (max - min)``, as in scipy.

    An invalid parameterisation (``max <= min``, a ``mode`` outside ``[min, max]``, a non-finite parameter, or a width
    ``max - min`` overflowing ``float64``) is not checked at construction; it raises ``InvalidOperation`` (a
    ``ComputeError``) when any method is evaluated. A null parameter nulls every method.
    """

    _min: pl.Expr
    _mode: pl.Expr
    _max: pl.Expr
    _distribution_name: ClassVar[DistributionName] = "triangular"

    def __init__(
        self,
        min: float | IntoExprColumn,  # noqa: A002
        mode: float | IntoExprColumn,
        max: float | IntoExprColumn,  # noqa: A002
    ) -> None:
        self._min = coerce_param(min, name="min")
        self._mode = coerce_param(mode, name="mode")
        self._max = coerce_param(max, name="max")
        self._scalar_kwargs = scalar_kwargs(min=scalar_float(min), mode=scalar_float(mode), max=scalar_float(max))

    @property
    def _param_exprs(self) -> tuple[pl.Expr, ...]:
        return (self._min, self._mode, self._max)

    @property
    def _validated_params(self) -> pl.Expr:
        return self._validated("max", self._max)

    @property
    def _width(self) -> pl.Expr:
        return self._max - self._min

    @property
    def _below_mode(self) -> pl.Expr:
        return self._mode - self._min

    @property
    def _above_mode(self) -> pl.Expr:
        return self._max - self._mode

    def mean(self) -> pl.Expr:
        """Expected value, ``(min + mode + max) / 3``, read off ``mode`` so a symmetric triangle's mean is ``mode``."""
        return self._moment(self._mode + (self._above_mode - self._below_mode) * _ONE_THIRD)

    def variance(self) -> pl.Expr:
        """Variance, ``(min**2 + mode**2 + max**2 - min * mode - min * max - mode * max) / 18``."""
        below, above = self._below_mode, self._above_mode
        return self._moment(
            below * (below * _ONE_EIGHTEENTH) + below * (above * _ONE_EIGHTEENTH) + above * (above * _ONE_EIGHTEENTH)
        )

    def std(self) -> pl.Expr:
        """Standard deviation, ``sqrt(variance)``.

        Not ``variance().sqrt()``: squaring and unsquaring the width saturates about 150 decades earlier.
        """
        below, above = self._below_mode, self._above_mode
        # Not `self._width`: with literal ends it is a literal, which polars divides by as a reciprocal multiply.
        width = below + above
        return self._moment(width * (1.0 - (below / width) * (above / width)).sqrt() * _ROOT_ONE_EIGHTEENTH)

    def median(self) -> pl.Expr:
        """Median, read from the end whose piece holds half the mass.

        ``min + sqrt((max - min) * (mode - min) / 2)`` with ``mode`` in the upper half, else
        ``max - sqrt((max - min) * (max - mode) / 2)``.
        """
        half_width = self._width * 0.5
        return self._moment(
            pl.when(self._below_mode >= half_width)
            .then(self._min + half_width.sqrt() * self._below_mode.sqrt())
            .otherwise(self._max - half_width.sqrt() * self._above_mode.sqrt())
        )

    def entropy(self) -> pl.Expr:
        """Differential entropy, ``1 / 2 + log((max - min) / 2)``."""
        return self._moment(self._width.log() + _HALF_MINUS_LN_2)
