from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar

import polars as pl

from polars_stats.distributions._base import ContinuousDistribution, coerce_param, scalar_float, scalar_kwargs

if TYPE_CHECKING:
    from polars_stats._typing import DistributionName, IntoExprColumn

_MEAN_DIVERGES_AT = 1
_VARIANCE_DIVERGES_AT = 2
"""Largest `shape` at which the mean, respectively the variance, integral still diverges."""


class Pareto(ContinuousDistribution):
    """Pareto (type I) distribution with scale ``scale`` and shape ``shape``, on the support ``x >= scale``.

    Equivalent to ``scipy.stats.pareto(b=shape, scale=scale)``: scipy's shape ``b`` is ``shape`` here, and the two
    arguments are in the opposite order, so pass both by keyword.

    Arguments:
        scale: Scale parameter ``x_m``, the lower bound of the support and the mode, with ``scale > 0``. Either a
            Python ``float`` or an ``IntoExprColumn`` (``pl.Expr``, ``pl.Series`` or column name ``str``) carrying one
            scale per row.
        shape: Shape parameter ``alpha``, the tail index (``sf(x) = (scale / x) ** shape``), with ``shape > 0``. Same
            accepted types as ``scale``.

    ``mean()`` is **``+inf``** for ``shape <= 1`` and ``variance()`` / ``std()`` for ``shape <= 2``, where the
    integral diverges, as in scipy.

    An invalid parameterisation (``scale <= 0``, ``shape <= 0`` or a non-finite parameter) is not checked at
    construction; it raises ``InvalidOperation`` (a ``ComputeError``) when any method is evaluated. Below the support
    ``pdf`` and ``cdf`` are ``0`` and ``sf`` is ``1``. A null parameter nulls every method, on the support and below it.
    """

    _scale: pl.Expr
    _shape: pl.Expr
    _distribution_name: ClassVar[DistributionName] = "pareto"

    def __init__(self, scale: float | IntoExprColumn, shape: float | IntoExprColumn) -> None:
        self._scale = coerce_param(scale, name="scale")
        self._shape = coerce_param(shape, name="shape")
        self._scalar_kwargs = scalar_kwargs(scale=scalar_float(scale), shape=scalar_float(shape))

    @property
    def _param_exprs(self) -> tuple[pl.Expr, ...]:
        return (self._scale, self._shape)

    @property
    def _validated_params(self) -> pl.Expr:
        return self._validated("shape", self._shape)

    def mean(self) -> pl.Expr:
        """Expected value, ``shape * scale / (shape - 1)`` for ``shape > 1``, else ``+inf``."""
        finite = self._scale * (self._shape / (self._shape - 1))
        return self._moment(pl.when(self._shape > _MEAN_DIVERGES_AT).then(finite).otherwise(math.inf))

    def variance(self) -> pl.Expr:
        """Variance, ``scale**2 * shape / ((shape - 1)**2 * (shape - 2))`` for ``shape > 2``, else ``+inf``."""
        finite = (self._scale / (self._shape - 1)) ** 2 * (self._shape / (self._shape - 2))
        return self._moment(pl.when(self._shape > _VARIANCE_DIVERGES_AT).then(finite).otherwise(math.inf))

    def std(self) -> pl.Expr:
        """Standard deviation, ``scale / (shape - 1) * sqrt(shape / (shape - 2))`` for ``shape > 2``, else ``+inf``.

        Not ``variance().sqrt()``: squaring and unsquaring the scale saturates about 300 decades earlier.
        """
        finite = self._scale / (self._shape - 1) * (self._shape / (self._shape - 2)).sqrt()
        return self._moment(pl.when(self._shape > _VARIANCE_DIVERGES_AT).then(finite).otherwise(math.inf))

    def entropy(self) -> pl.Expr:
        """Differential entropy, ``log(scale / shape) + 1 / shape + 1``.

        Summed as ``log(scale) - log(shape)``: the ratio over- and underflows where neither log does.
        """
        return self._moment(self._scale.log() - self._shape.log() + 1 / self._shape + 1)
