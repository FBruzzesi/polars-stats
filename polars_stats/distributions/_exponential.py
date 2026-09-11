from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar

from polars_stats.distributions._base import ContinuousDistribution, coerce_param, scalar_float, scalar_kwargs

if TYPE_CHECKING:
    import polars as pl

    from polars_stats._typing import DistributionName, IntoExprColumn


class Exponential(ContinuousDistribution):
    """Exponential distribution with rate ``rate`` (λ).

    Equivalent to ``scipy.stats.expon(scale=1 / rate)``. The API exposes ``rate`` (the ``statrs``
    parameterisation) rather than scipy's ``scale = 1 / rate``: it is the natural parameter and
    avoids the divide-by-zero footgun of passing ``scale=0``.

    Arguments:
        rate: Rate parameter λ, with ``rate > 0``. Either a Python ``float`` or an ``IntoExprColumn``
            (``pl.Expr``, ``pl.Series`` or column name ``str``) carrying one rate per row.

    An invalid ``rate`` (``rate <= 0`` or ``NaN``) is not checked at construction; it raises
    ``InvalidOperation`` (a ``ComputeError``) when any method is evaluated. The support is ``x >= 0``:
    ``pdf`` and ``cdf`` are ``0`` for ``x < 0``, and ``sf`` is ``1`` there. A null ``rate`` nulls every
    method, on the support and below it.
    """

    _rate: pl.Expr
    _distribution_name: ClassVar[DistributionName] = "exponential"

    def __init__(self, rate: float | IntoExprColumn) -> None:
        self._rate = coerce_param(rate, name="rate")
        self._scalar_kwargs = scalar_kwargs(rate=scalar_float(rate))

    @property
    def _param_exprs(self) -> tuple[pl.Expr, ...]:
        return (self._rate,)

    @property
    def _validated_params(self) -> pl.Expr:
        """``rate`` validated in Rust to be strictly positive; every moment names it once."""
        return self._validated("rate", self._rate)

    def mean(self) -> pl.Expr:
        """Expected value, ``1 / rate``."""
        return 1 / self._validated_params

    def variance(self) -> pl.Expr:
        """Variance, ``1 / rate**2``."""
        return 1 / self._validated_params**2

    def std(self) -> pl.Expr:
        """Standard deviation, ``1 / rate``.

        Not ``variance().sqrt()``: squaring and unsquaring the rate saturates about 300 decades earlier.
        """
        return 1 / self._validated_params

    def median(self) -> pl.Expr:
        """Median, ``log(2) / rate``."""
        return math.log(2) / self._validated_params

    def entropy(self) -> pl.Expr:
        """Differential entropy, ``1 - log(rate)``."""
        return 1 - self._validated_params.log()
