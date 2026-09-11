from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar

from polars_stats.distributions._base import (
    ContinuousDistribution,
    coerce_param,
    expm1,
    log_abs_expm1,
    scalar_float,
    scalar_kwargs,
)

if TYPE_CHECKING:
    import polars as pl

    from polars_stats._typing import DistributionName, IntoExprColumn


class LogNormal(ContinuousDistribution):
    """Log-normal distribution: ``X`` such that ``ln(X)`` is ``Normal(mu, sigma)``.

    Equivalent to ``scipy.stats.lognorm(s=sigma, scale=exp(mu))`` (with ``loc=0``): scipy's shape ``s`` is
    ``sigma`` and its ``scale`` is ``exp(mu)``.

    The support is ``x > 0``; ``pdf`` and ``cdf`` are ``0`` and ``sf`` is ``1`` for ``x <= 0``, matching scipy.

    Arguments:
        mu: Location of the underlying normal (mean of ``ln(X)``). Either a Python ``float`` or an
            ``IntoExprColumn`` (``pl.Expr``, ``pl.Series`` or column name ``str``) carrying one value per row.
        sigma: Scale of the underlying normal (std-dev of ``ln(X)``), with ``sigma > 0``. Same accepted types as ``mu``.

    An invalid parameterisation (``sigma <= 0`` or a non-finite parameter) is not checked at construction;
    it raises ``InvalidOperation`` (a ``ComputeError``) when any method is evaluated. Null parameters propagate
    to null.
    """

    _mu: pl.Expr
    _sigma: pl.Expr
    _distribution_name: ClassVar[DistributionName] = "lognormal"

    def __init__(self, mu: float | IntoExprColumn = 0.0, sigma: float | IntoExprColumn = 1.0) -> None:
        self._mu = coerce_param(mu, name="mu")
        self._sigma = coerce_param(sigma, name="sigma")
        self._scalar_kwargs = scalar_kwargs(mu=scalar_float(mu), sigma=scalar_float(sigma))

    @property
    def _param_exprs(self) -> tuple[pl.Expr, ...]:
        return (self._mu, self._sigma)

    @property
    def _validated_params(self) -> pl.Expr:
        return self._validated("sigma", self._sigma)

    @property
    def _half_sigma_sq(self) -> pl.Expr:
        return self._sigma**2 / 2

    def mean(self) -> pl.Expr:
        """Expected value, ``exp(mu + sigma ** 2 / 2)``."""
        return self._moment((self._mu + self._half_sigma_sq).exp())

    def variance(self) -> pl.Expr:
        """Variance, ``(exp(sigma ** 2) - 1) * exp(2 * mu + sigma ** 2)``.

        The leading factor goes through `expm1`: the literal ``exp(sigma ** 2) - 1`` cancels for a small
        ``sigma``. No cut-over is needed, the argument is positive.
        """
        return self._moment(expm1(self._sigma**2) * (2 * self._mu + self._sigma**2).exp())

    def std(self) -> pl.Expr:
        """Standard deviation, ``exp(0.5 * log(exp(sigma ** 2) - 1) + mu + sigma ** 2 / 2)``.

        Not ``variance().sqrt()``: the variance overflows ``f64`` above ``sigma ~ 18.8`` (so ``inf`` is right
        *there*), the standard deviation only above ``sigma ~ 26.6``. ``std() ** 2`` and ``variance()`` are
        therefore not interchangeable at a large ``sigma``.
        """
        return self._moment((0.5 * log_abs_expm1(self._sigma**2) + self._mu + self._half_sigma_sq).exp())

    def median(self) -> pl.Expr:
        """Median, ``exp(mu)``."""
        return self._moment(self._mu.exp())

    def entropy(self) -> pl.Expr:
        """Differential entropy, ``mu + 0.5 * log(2 * pi * e * sigma ** 2)``."""
        return self._moment(self._mu + 0.5 * (math.tau * math.e * self._sigma**2).log())
