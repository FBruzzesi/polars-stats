from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar

from polars_stats.distributions._base import (
    ContinuousDistribution,
    coerce_param,
    scalar_float,
    scalar_kwargs,
)

if TYPE_CHECKING:
    import polars as pl

    from polars_stats._typing import IntoExprColumn

_TWO_PI_E = math.tau * math.e
"""2 * pi * e, the constant inside the normal's differential entropy `0.5 * log(2 * pi * e * sigma^2)`"""


class Normal(ContinuousDistribution):
    """Normal (Gaussian) distribution with location ``mu`` and scale ``sigma``.

    Equivalent to ``scipy.stats.norm(loc=mu, scale=sigma)``. The standard normal is the default parameterisation
    (``mu=0``, ``sigma=1``); it is handled by the same code path as any other, not special-cased.

    Arguments:
        mu: Location parameter. Either a Python ``float`` or an ``IntoExprColumn`` (``pl.Expr``, ``pl.Series`` or
            column name ``str``) carrying one location per row.
        sigma: Scale parameter, with ``sigma > 0``. Same accepted types as ``mu``.

    An invalid scale (``sigma <= 0`` or a non-finite parameter) is not checked at construction;
    it raises ``InvalidOperation`` (a ``ComputeError``) when any method is evaluated.

    Null parameters propagate to null.
    """

    _mu: pl.Expr
    _sigma: pl.Expr
    _plugin_prefix: ClassVar[str] = "normal"

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
    def _checked_params(self) -> pl.Expr:
        """``sigma`` validated in Rust against the full ``(mu, sigma)`` parameterisation.

        See ``_UnivariateDistribution._checked_params`` / ``_checked`` for the moment-gating contract.
        """
        return self._checked("normal_sigma", self._sigma)

    def _pdf(self, value: pl.Expr) -> pl.Expr:
        """Density via native ``statrs`` ``Continuous::pdf``."""
        return self._value_plugin("normal_pdf", value)

    def _log_pdf(self, value: pl.Expr) -> pl.Expr:
        """Log-density via native ``Continuous::ln_pdf`` (more accurate than ``pdf().log()``)."""
        return self._value_plugin("normal_ln_pdf", value)

    def _cdf(self, value: pl.Expr) -> pl.Expr:
        """Cumulative distribution via native ``ContinuousCDF::cdf``."""
        return self._value_plugin("normal_cdf", value)

    def _log_cdf(self, value: pl.Expr) -> pl.Expr:
        """Log-cdf via the native stable ``ln_erfc`` form (finite past ~38 sigma, unlike ``cdf().log()``)."""
        return self._value_plugin("normal_ln_cdf", value)

    def _sf(self, value: pl.Expr) -> pl.Expr:
        """Survival function via native ``ContinuousCDF::sf`` (accurate in the upper tail).

        ``isf`` inherits the base-class default ``ppf(1 - quantile)`` over the native ``ppf``.
        """
        return self._value_plugin("normal_sf", value)

    def _log_sf(self, value: pl.Expr) -> pl.Expr:
        """Log-sf via the native stable ``ln_erfc`` form (finite past ~38 sigma, unlike ``sf().log()``).

        The flagship anomaly-scoring path: ``sf`` for a many-sigma event underflows to ``0`` and its log to
        ``-inf``; this stays finite (``scipy.stats.norm.logsf``-equivalent).
        """
        return self._value_plugin("normal_ln_sf", value)

    def _ppf(self, quantile: pl.Expr) -> pl.Expr:
        """Inverse cdf via the closed-form ``ContinuousCDF::inverse_cdf``.

        A quantile outside ``[0, 1]`` yields null; the endpoints map to the infinite tails
        (``ppf(0) = -inf``, ``ppf(1) = +inf``), matching scipy.
        """
        return self._value_plugin("normal_ppf", quantile)

    def mean(self) -> pl.Expr:
        """Expected value, the ``mu`` location parameter."""
        return self._moment(self._mu)

    def variance(self) -> pl.Expr:
        """Variance, ``sigma ** 2``."""
        return self._moment(self._sigma**2)

    def median(self) -> pl.Expr:
        """Median, equal to the ``mu`` location parameter."""
        return self._moment(self._mu)

    def entropy(self) -> pl.Expr:
        """Differential entropy, ``0.5 * log(2 * pi * e * sigma ** 2)``."""
        return self._moment(0.5 * (_TWO_PI_E * self._sigma**2).log())
