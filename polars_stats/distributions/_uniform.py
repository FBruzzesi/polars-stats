from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar

from polars_stats.distributions._base import ContinuousDistribution, coerce_param, scalar_float, scalar_kwargs

if TYPE_CHECKING:
    import polars as pl

    from polars_stats._typing import DistributionName, IntoExprColumn


class Uniform(ContinuousDistribution):
    """Continuous uniform distribution over ``[min, max]``.

    Equivalent to ``scipy.stats.uniform(loc=min, scale=max - min)``.

    Following scipy, the density, cdf and the other closed forms treat the support as the closed interval ``[min, max]``
    (so ``pdf(max) == 1 / (max - min)``); the ``sample`` plugin draws on the half-open ``[min, max)``.

    Arguments:
        min: Lower bound. Either a Python ``float`` or an ``IntoExprColumn`` (``pl.Expr``,
            ``pl.Series`` or column name ``str``) carrying one bound per row.
        max: Upper bound, with ``max > min``. Same accepted types as ``min``.

    An invalid parameterisation (``max <= min``, a non-finite bound, or a width ``max - min`` overflowing
    ``float64``) is not checked at construction; it raises ``InvalidOperation`` (a ``ComputeError``) when any
    method is evaluated. A null bound nulls every method, on the support and off it.
    """

    _min: pl.Expr
    _max: pl.Expr
    _distribution_name: ClassVar[DistributionName] = "uniform"

    def __init__(self, min: float | IntoExprColumn, max: float | IntoExprColumn) -> None:  # noqa: A002
        self._min = coerce_param(min, name="min")
        self._max = coerce_param(max, name="max")
        self._scalar_kwargs = scalar_kwargs(min=scalar_float(min), max=scalar_float(max))

    @property
    def _param_exprs(self) -> tuple[pl.Expr, ...]:
        return (self._min, self._max)

    @property
    def _validated_params(self) -> pl.Expr:
        return self._validated("range", self._max - self._min)

    @property
    def range(self) -> pl.Expr:
        """Width of the support, ``max - min``, validated in Rust; the moments derive from it."""
        return self._validated_params

    def mean(self) -> pl.Expr:
        """Expected value, ``(min + max) / 2``."""
        return self._min + self.range / 2

    def variance(self) -> pl.Expr:
        """Variance, ``(max - min)^2 / 12``."""
        return self.range**2 / 12

    def std(self) -> pl.Expr:
        """Standard deviation, ``(max - min) / sqrt(12)``.

        Not ``variance().sqrt()``: squaring and unsquaring the span saturates about 300 decades earlier.
        """
        return self.range / math.sqrt(12)

    def median(self) -> pl.Expr:
        """Median, ``(min + max) / 2``."""
        return self._min + self.range / 2

    def entropy(self) -> pl.Expr:
        """Differential entropy, ``log(max - min)``."""
        return self.range.log()
