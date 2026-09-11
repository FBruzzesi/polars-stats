from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar

from polars_stats.distributions._base import ContinuousDistribution, coerce_param, scalar_float, scalar_kwargs

if TYPE_CHECKING:
    import polars as pl

    from polars_stats._typing import DistributionName, IntoExprColumn


class Normal(ContinuousDistribution):
    """Normal (Gaussian) distribution with location ``mu`` and scale ``sigma``.

    Equivalent to ``scipy.stats.norm(loc=mu, scale=sigma)``. The standard normal (``mu=0``, ``sigma=1``) is the
    default parameterisation.

    Arguments:
        mu: Location parameter. Either a Python ``float`` or an ``IntoExprColumn`` (``pl.Expr``, ``pl.Series`` or
            column name ``str``) carrying one location per row.
        sigma: Scale parameter, with ``sigma > 0``. Same accepted types as ``mu``.

    An invalid scale (``sigma <= 0`` or a non-finite parameter) is not checked at construction; it raises
    ``InvalidOperation`` (a ``ComputeError``) when any method is evaluated. Null parameters propagate to null.
    """

    _mu: pl.Expr
    _sigma: pl.Expr
    _distribution_name: ClassVar[DistributionName] = "normal"

    def __init__(
        self,
        mu: float | IntoExprColumn = 0.0,
        sigma: float | IntoExprColumn = 1.0,
    ) -> None:
        self._mu = coerce_param(mu, name="mu")
        self._sigma = coerce_param(sigma, name="sigma")
        self._scalar_kwargs = scalar_kwargs(mu=scalar_float(mu), sigma=scalar_float(sigma))

    @property
    def _param_exprs(self) -> tuple[pl.Expr, ...]:
        return (self._mu, self._sigma)

    @property
    def _validated_params(self) -> pl.Expr:
        return self._validated("sigma", self._sigma)

    def mean(self) -> pl.Expr:
        """Expected value, ``mu``."""
        return self._moment(self._mu)

    def variance(self) -> pl.Expr:
        """Variance, ``sigma ** 2``."""
        return self._moment(self._sigma**2)

    def std(self) -> pl.Expr:
        """Standard deviation, ``sigma``.

        Not ``variance().sqrt()``: squaring and unsquaring ``sigma`` saturates across roughly 300 decades.
        """
        return self._moment(self._sigma)

    def median(self) -> pl.Expr:
        """Median, ``mu``."""
        return self._moment(self._mu)

    def entropy(self) -> pl.Expr:
        """Differential entropy, ``0.5 * log(2 * pi * e * sigma ** 2)``."""
        return self._moment(0.5 * (math.tau * math.e * self._sigma**2).log())
