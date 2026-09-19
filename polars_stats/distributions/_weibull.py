from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from polars_stats.distributions._base import ContinuousDistribution, coerce_param, scalar_float, scalar_kwargs

if TYPE_CHECKING:
    import polars as pl

    from polars_stats._typing import DistributionName, IntoExprColumn

_EULER_GAMMA = 0.5772156649015329
"""The Euler-Mascheroni constant, `-digamma(1)`."""


class Weibull(ContinuousDistribution):
    """Weibull distribution with shape ``shape`` and scale ``scale``, on the support ``x >= 0``.

    Equivalent to ``scipy.stats.weibull_min(c=shape, scale=scale)``: scipy's shape ``c`` is ``shape`` here.
    ``Weibull(shape=1.0, scale=s)`` is ``Exponential(rate=1 / s)``.

    Arguments:
        shape: Shape parameter ``k``, with ``shape > 0``. Either a Python ``float`` or an ``IntoExprColumn``
            (``pl.Expr``, ``pl.Series`` or column name ``str``) carrying one shape per row.
        scale: Scale parameter ``lambda``, with ``scale > 0``. Same accepted types as ``shape``.

    An invalid parameterisation (``shape <= 0``, ``scale <= 0`` or a non-finite parameter) is not checked at
    construction; it raises ``InvalidOperation`` (a ``ComputeError``) when any method is evaluated. Below the support
    ``pdf`` and ``cdf`` are ``0`` and ``sf`` is ``1``. A null parameter nulls every method, on the support and below it.
    """

    _shape: pl.Expr
    _scale: pl.Expr
    _distribution_name: ClassVar[DistributionName] = "weibull"

    def __init__(self, shape: float | IntoExprColumn, scale: float | IntoExprColumn) -> None:
        self._shape = coerce_param(shape, name="shape")
        self._scale = coerce_param(scale, name="scale")
        self._scalar_kwargs = scalar_kwargs(shape=scalar_float(shape), scale=scalar_float(scale))

    @property
    def _param_exprs(self) -> tuple[pl.Expr, ...]:
        return (self._shape, self._scale)

    @property
    def _validated_params(self) -> pl.Expr:
        return self._validated("scale", self._scale)

    def mean(self) -> pl.Expr:
        """Expected value, ``scale * Gamma(1 + 1 / shape)``."""
        return self._param_plugin("mean")

    def variance(self) -> pl.Expr:
        """Variance, ``scale**2 * (Gamma(1 + 2 / shape) - Gamma(1 + 1 / shape)**2)``."""
        return self._param_plugin("variance")

    def std(self) -> pl.Expr:
        """Standard deviation, ``scale * sqrt(Gamma(1 + 2 / shape) - Gamma(1 + 1 / shape)**2)``.

        Not ``variance().sqrt()``: squaring and unsquaring the scale saturates about 300 decades earlier.
        """
        return self._param_plugin("std")

    def entropy(self) -> pl.Expr:
        """Differential entropy, ``euler_gamma * (1 - 1 / shape) + log(scale / shape) + 1``.

        Summed as ``log(scale) - log(shape)``: the ratio over- and underflows where neither log does.
        """
        return self._moment(_EULER_GAMMA * (1 - 1 / self._shape) + self._scale.log() - self._shape.log() + 1)
