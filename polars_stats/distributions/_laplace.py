from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar

from polars_stats.distributions._base import ContinuousDistribution, coerce_param, scalar_float, scalar_kwargs

if TYPE_CHECKING:
    import polars as pl

    from polars_stats._typing import DistributionName, IntoExprColumn

_LOG_TWO_E = 1.0 + math.log(2.0)
"""`ln(2 e)`, summed with `log(scale)`: the product `2 e scale` overflows above `scale ~ 3.3e307`, where the entropy is
still only 709.8."""


class Laplace(ContinuousDistribution):
    """Laplace (double exponential) distribution with location ``loc`` and scale ``scale``.

    Equivalent to ``scipy.stats.laplace(loc=loc, scale=scale)``.

    Arguments:
        loc: Location parameter, the mean, median and mode. Either a Python ``float`` or an ``IntoExprColumn``
            (``pl.Expr``, ``pl.Series`` or column name ``str``) carrying one location per row.
        scale: Scale parameter, with ``scale > 0``. Same accepted types as ``loc``.

    An invalid scale (``scale <= 0`` or a non-finite parameter) is not checked at construction; it raises
    ``InvalidOperation`` (a ``ComputeError``) when any method is evaluated. Null parameters propagate to null.
    """

    _loc: pl.Expr
    _scale: pl.Expr
    _distribution_name: ClassVar[DistributionName] = "laplace"

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
        """Expected value, ``loc``."""
        return self._moment(self._loc)

    def variance(self) -> pl.Expr:
        """Variance, ``2 * scale ** 2``."""
        return self._moment(self._scale**2 * 2.0)

    def std(self) -> pl.Expr:
        """Standard deviation, ``sqrt(2) * scale``.

        Not ``variance().sqrt()``: squaring and unsquaring ``scale`` saturates across roughly 300 decades.
        """
        return self._moment(self._scale * math.sqrt(2.0))

    def median(self) -> pl.Expr:
        """Median, ``loc``."""
        return self._moment(self._loc)

    def entropy(self) -> pl.Expr:
        """Differential entropy, ``log(2 * e * scale)``."""
        return self._moment(_LOG_TWO_E + self._scale.log())
