from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar

import polars as pl

from polars_stats.distributions._base import ContinuousDistribution, coerce_param, scalar_float, scalar_kwargs

if TYPE_CHECKING:
    from polars_stats._typing import DistributionName, IntoExprColumn

_UNDEFINED = pl.lit(None, dtype=pl.Float64)
"""A moment the distribution does not have. The explicit dtype keeps the output schema `Float64` rather than `Null`."""

_LOG_FOUR_PI = math.log(4 * math.pi)
"""Added to `log(scale)` rather than multiplied into it: `log(4 pi scale)` overflows above `scale ~ 1.43e307`, where
the true entropy is only 709.8. Splitting the log is also how scipy spells it."""


class Cauchy(ContinuousDistribution):
    """Cauchy distribution with location ``loc`` and scale ``scale``.

    Equivalent to ``scipy.stats.cauchy(loc=loc, scale=scale)``.

    Arguments:
        loc: Location parameter, the median and the mode. Either a Python ``float`` or an ``IntoExprColumn``
            (``pl.Expr``, ``pl.Series`` or column name ``str``) carrying one location per row.
        scale: Scale parameter, the half-width at half-maximum, with ``scale > 0``. Same accepted types as ``loc``.

    Cauchy has no moments of any order, so ``mean()``, ``variance()`` and ``std()`` are **null** where scipy
    returns ``nan``. All three still validate: an invalid ``scale`` raises rather than nulling.

    An invalid scale (``scale <= 0`` or a non-finite parameter) is not checked at construction; it raises
    ``InvalidOperation`` (a ``ComputeError``) when any method is evaluated. Null parameters propagate to null.
    """

    _loc: pl.Expr
    _scale: pl.Expr
    _distribution_name: ClassVar[DistributionName] = "cauchy"

    def __init__(self, loc: float | IntoExprColumn, scale: float | IntoExprColumn) -> None:
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
        """Undefined, so **null** on every valid row; an invalid ``scale`` still raises."""
        return self._moment(_UNDEFINED)

    def variance(self) -> pl.Expr:
        """Undefined, so **null** on every valid row; an invalid ``scale`` still raises."""
        return self._moment(_UNDEFINED)

    def median(self) -> pl.Expr:
        """Median, ``loc``."""
        return self._moment(self._loc)

    def entropy(self) -> pl.Expr:
        """Differential entropy, ``log(4 * pi * scale)``."""
        return self._moment(_LOG_FOUR_PI + self._scale.log())
