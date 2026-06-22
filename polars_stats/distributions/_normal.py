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
"""2 * pi * e, the constant inside the normal's differential entropy `0.5 * log(2 * pi * e * std_dev^2)`"""


class Normal(ContinuousDistribution):
    """Normal (Gaussian) distribution with location ``mean`` and scale ``std_dev``.

    Equivalent to ``scipy.stats.norm(loc=mean, scale=std_dev)``. The standard normal is the default parameterisation
    (``mean=0``, ``std_dev=1``); it is handled by the same code path as any other, not special-cased.

    Arguments:
        mean: Location parameter. Either a Python ``float`` or an ``IntoExprColumn`` (``pl.Expr``, ``pl.Series`` or
            column name ``str``) carrying one location per row.
        std_dev: Scale parameter, with ``std_dev > 0``. Same accepted types as ``mean``.

    An invalid scale (``std_dev <= 0`` or a non-finite parameter) is not checked at construction;
    it raises ``InvalidOperation`` (a ``ComputeError``) when any method is evaluated.

    Null parameters propagate to null.
    """

    _mean: pl.Expr
    _std_dev: pl.Expr
    _plugin_prefix: ClassVar[str] = "normal"

    def __init__(
        self,
        mean: float | IntoExprColumn = 0.0,
        std_dev: float | IntoExprColumn = 1.0,
    ) -> None:
        self._mean = coerce_param(mean, name="mean")
        self._std_dev = coerce_param(std_dev, name="std_dev")
        # Constant parameters enable the fast sampler path; `None` falls back to the per-row plugin.
        self._scalar_kwargs = scalar_kwargs(mean=scalar_float(mean), std_dev=scalar_float(std_dev))

    @property
    def _param_exprs(self) -> tuple[pl.Expr, ...]:
        return (self._mean, self._std_dev)

    @property
    def _checked_params(self) -> pl.Expr:
        """``std_dev`` validated in Rust against the full ``(mean, std_dev)`` parameterisation."""
        # Mirrors ``Uniform.range`` / ``Bernoulli._checked_p``: the closed-form moments derive from this
        # FFI round-trip, so they report an invalid parameterisation (``std_dev <= 0``, or a non-finite parameter) as a
        # ``ComputeError`` consistently with the value-keyed methods. Null in either parameter propagates to null, so a
        # moment built on this nulls when either input is null.
        # `_checked` validates once for scalar parameters (length-1 inputs) and per-row for columns.
        return self._checked("normal_std_dev", self._std_dev)

    def _pdf(self, value: pl.Expr) -> pl.Expr:
        """Density via native ``statrs`` ``Continuous::pdf``."""
        return self._value_plugin("normal_pdf", value)

    def _log_pdf(self, value: pl.Expr) -> pl.Expr:
        """Log-density via native ``Continuous::ln_pdf`` (more accurate than ``pdf().log()``)."""
        return self._value_plugin("normal_ln_pdf", value)

    def _cdf(self, value: pl.Expr) -> pl.Expr:
        """Cumulative distribution via native ``ContinuousCDF::cdf``."""
        return self._value_plugin("normal_cdf", value)

    def _sf(self, value: pl.Expr) -> pl.Expr:
        """Survival function via native ``ContinuousCDF::sf`` (accurate in the upper tail).

        ``log_sf`` and ``isf`` inherit the base-class defaults, which compose this native ``sf`` and
        ``ppf`` rather than the generic ``1 - cdf`` fallback.
        """
        return self._value_plugin("normal_sf", value)

    def _ppf(self, quantile: pl.Expr) -> pl.Expr:
        """Inverse cdf via the closed-form ``ContinuousCDF::inverse_cdf``.

        A quantile outside ``[0, 1]`` yields null; the endpoints map to the infinite tails
        (``ppf(0) = -inf``, ``ppf(1) = +inf``), matching scipy.
        """
        return self._value_plugin("normal_ppf", quantile)

    def mean(self) -> pl.Expr:
        """Expected value, the ``mean`` location parameter."""
        return self._moment(self._mean)

    def variance(self) -> pl.Expr:
        """Variance, ``std_dev ** 2``."""
        return self._moment(self._std_dev**2)

    def median(self) -> pl.Expr:
        """Median, equal to the ``mean`` location parameter."""
        return self._moment(self._mean)

    def entropy(self) -> pl.Expr:
        """Differential entropy, ``0.5 * log(2 * pi * e * std_dev ** 2)``."""
        return self._moment(0.5 * (_TWO_PI_E * self._std_dev**2).log())
