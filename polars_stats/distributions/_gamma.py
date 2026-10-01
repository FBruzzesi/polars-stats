from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from polars_stats.distributions._base import ContinuousDistribution, coerce_param, scalar_float, scalar_kwargs

if TYPE_CHECKING:
    import polars as pl

    from polars_stats._typing import DistributionName, IntoExprColumn


class Gamma(ContinuousDistribution):
    """Gamma distribution with shape ``shape`` and rate ``rate``, on the support ``x >= 0``.

    Equivalent to ``scipy.stats.gamma(a=shape, scale=1 / rate)``. It takes the ``statrs`` ``rate``, as
    ``Exponential`` does, so ``Gamma(shape=1.0, rate=r)`` is ``Exponential(rate=r)``.

    Arguments:
        shape: Shape parameter ``alpha``, with ``shape > 0``. Either a Python ``float`` or an ``IntoExprColumn``
            (``pl.Expr``, ``pl.Series`` or column name ``str``) carrying one shape per row.
        rate: Rate parameter ``beta``, with ``rate > 0``. Same accepted types as ``shape``.

    An invalid parameterisation (``shape <= 0``, ``rate <= 0`` or a non-finite parameter) is not checked at
    construction; it raises ``InvalidOperation`` (a ``ComputeError``) when any method is evaluated. Below the support
    ``pdf`` and ``cdf`` are ``0`` and ``sf`` is ``1``. A null parameter nulls every method, on the support and below it.
    """

    _shape: pl.Expr
    _rate: pl.Expr
    _distribution_name: ClassVar[DistributionName] = "gamma"

    def __init__(self, shape: float | IntoExprColumn, rate: float | IntoExprColumn) -> None:
        self._shape = coerce_param(shape, name="shape")
        self._rate = coerce_param(rate, name="rate")
        self._scalar_kwargs = scalar_kwargs(shape=scalar_float(shape), rate=scalar_float(rate))

    @property
    def _param_exprs(self) -> tuple[pl.Expr, ...]:
        return (self._shape, self._rate)

    @property
    def _validated_params(self) -> pl.Expr:
        return self._validated("rate", self._rate)

    def mean(self) -> pl.Expr:
        """Expected value, ``shape / rate``."""
        return self._moment(self._shape / self._rate)

    def variance(self) -> pl.Expr:
        """Variance, ``shape / rate**2``.

        Squared from ``std()``: ``rate**2`` alone over- and underflows where the variance does not.
        """
        return self.std() ** 2

    def std(self) -> pl.Expr:
        """Standard deviation, ``sqrt(shape) / rate``.

        Not ``variance().sqrt()``, which saturates about 300 decades earlier.
        """
        return self._moment(self._shape.sqrt() / self._rate)

    def entropy(self) -> pl.Expr:
        """Differential entropy, ``shape - log(rate) + log Gamma(shape) + (1 - shape) * digamma(shape)``.

        Digamma has no Polars expression, so ``gamma_entropy`` evaluates it in Rust.
        """
        return self._param_plugin("entropy")
