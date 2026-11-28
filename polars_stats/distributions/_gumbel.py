from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar

from polars_stats.distributions._base import (
    EULER_GAMMA,
    ContinuousDistribution,
    coerce_param,
    scalar_float,
    scalar_kwargs,
)

if TYPE_CHECKING:
    import polars as pl

    from polars_stats._typing import DistributionName, IntoExprColumn

_STANDARD_MEDIAN = -math.log(math.log(2.0))
"""`-ln(ln 2)`, the median of `Gumbel(0, 1)`."""

_STANDARD_VARIANCE = math.pi**2 / 6.0
"""`pi^2 / 6`, the variance of `Gumbel(0, 1)`."""

_STANDARD_STD = math.pi / math.sqrt(6.0)
"""`pi / sqrt(6)`, the standard deviation of `Gumbel(0, 1)`."""


class Gumbel(ContinuousDistribution):
    """Gumbel (maximum extreme value, type I) distribution with location ``loc`` and scale ``scale``.

    Equivalent to ``scipy.stats.gumbel_r(loc=loc, scale=scale)``. ``scipy.stats.gumbel_l(loc, scale)``, the minimum
    variant, is ``-X`` for ``X ~ Gumbel(loc=-loc, scale=scale)``.

    Arguments:
        loc: Location parameter, the mode. Either a Python ``float`` or an ``IntoExprColumn`` (``pl.Expr``,
            ``pl.Series`` or column name ``str``) carrying one location per row.
        scale: Scale parameter, with ``scale > 0``. Same accepted types as ``loc``.

    An invalid scale (``scale <= 0`` or a non-finite parameter) is not checked at construction; it raises
    ``InvalidOperation`` (a ``ComputeError``) when any method is evaluated. Null parameters propagate to null.
    """

    _loc: pl.Expr
    _scale: pl.Expr
    _distribution_name: ClassVar[DistributionName] = "gumbel"

    def __init__(self, loc: float | IntoExprColumn = 0.0, scale: float | IntoExprColumn = 1.0) -> None:
        self._loc = coerce_param(loc, name="loc")
        self._scale = coerce_param(scale, name="scale")
        self._scalar_kwargs = scalar_kwargs(loc=scalar_float(loc), scale=scalar_float(scale))

    @property
    def _param_exprs(self) -> tuple[pl.Expr, ...]:
        return (self._loc, self._scale)

    @property
    def _validated_params(self) -> pl.Expr:
        return self._validated("scale", self._scale)

    def mean(self) -> pl.Expr:
        """Expected value, ``loc + euler_gamma * scale``."""
        return self._moment(self._loc + self._scale * EULER_GAMMA)

    def variance(self) -> pl.Expr:
        """Variance, ``pi ** 2 / 6 * scale ** 2``."""
        return self._moment(self._scale**2 * _STANDARD_VARIANCE)

    def std(self) -> pl.Expr:
        """Standard deviation, ``pi / sqrt(6) * scale``.

        Not ``variance().sqrt()``: squaring and unsquaring ``scale`` saturates across roughly 300 decades.
        """
        return self._moment(self._scale * _STANDARD_STD)

    def median(self) -> pl.Expr:
        """Median, ``loc - scale * log(log(2))``."""
        return self._moment(self._loc + self._scale * _STANDARD_MEDIAN)

    def entropy(self) -> pl.Expr:
        """Differential entropy, ``log(scale) + euler_gamma + 1``."""
        return self._moment(self._scale.log() + (EULER_GAMMA + 1.0))
