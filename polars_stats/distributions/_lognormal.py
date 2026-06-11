from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar

import polars as pl

from polars_stats.distributions._base import (
    ROW_INDEX_EXPR,
    ContinuousDistribution,
    coerce_param,
    register_plugin,
    scalar_float,
    scalar_kwargs,
)

if TYPE_CHECKING:
    from polars_stats._typing import IntoExprColumn, PolarsDataType


_TWO_PI_E = math.tau * math.e
"""2 * pi * e, the constant inside the log-normal's differential entropy `mu + 0.5 * log(2 * pi * e * sigma^2)`."""


class LogNormal(ContinuousDistribution):
    """Log-normal distribution: ``X`` such that ``ln(X)`` is ``Normal(mu, sigma)``.

    Parameterised by the underlying normal's location ``mu`` and scale ``sigma`` (``sigma > 0``).

    Equivalent to ``scipy.stats.lognorm(s=sigma, scale=exp(mu))`` (with ``loc=0``):
    scipy's shape ``s`` is ``sigma`` and its ``scale`` is ``exp(mu)``.

    The support is ``x > 0``; ``pdf`` and ``cdf`` are ``0`` and ``sf`` is ``1`` for ``x <= 0``, matching scipy.

    Arguments:
        mu: Location of the underlying normal (mean of ``ln(X)``). Either a Python ``float`` or an
            ``IntoExprColumn`` (``pl.Expr``, ``pl.Series`` or column name ``str``) carrying one value per row.
        sigma: Scale of the underlying normal (std-dev of ``ln(X)``), with ``sigma > 0``. Same accepted types as ``mu``.

    An invalid parameterisation (``sigma <= 0`` or a non-finite parameter) is not checked at construction;
    it raises ``InvalidOperation`` (a ``ComputeError``) when any method is evaluated.

    Null parameters propagate to null.
    """

    _mu: pl.Expr
    _sigma: pl.Expr
    _sample_dtype: ClassVar[PolarsDataType] = pl.Float64()

    def __init__(self, mu: float | IntoExprColumn = 0.0, sigma: float | IntoExprColumn = 1.0) -> None:
        self._mu = coerce_param(mu, name="mu")
        self._sigma = coerce_param(sigma, name="sigma")
        # Constant parameters enable the fast sampler path; `None` falls back to the per-row plugin.
        self._scalar_kwargs = scalar_kwargs(mu=scalar_float(mu), sigma=scalar_float(sigma))

    def _value_plugin(self, function_name: str, value: pl.Expr) -> pl.Expr:
        """Register a value-keyed Rust plugin call ``f(value, mu, sigma)``.

        Validation of ``sigma`` happens inside the plugin, so every value-keyed method reports an
        invalid parameterisation consistently; null inputs propagate per row.
        """
        return register_plugin(function_name, (value, self._mu, self._sigma))

    @property
    def _checked_sigma(self) -> pl.Expr:
        """``sigma`` validated in Rust against the full ``(mu, sigma)`` parameterisation."""
        # Mirrors ``Normal._checked_std_dev`` / ``Uniform.range``: the closed-form moments derive from
        # this single FFI round-trip, so they report an invalid parameterisation (``sigma <= 0``, or a
        # non-finite parameter) as a ``ComputeError`` consistently with the value-keyed methods. Null in
        # either parameter propagates to null, so a moment built on this nulls when either input is null.
        return register_plugin("lognormal_sigma", (self._mu, self._sigma))

    def _valid_mask(self) -> pl.Expr:
        # Only null parameters are masked to a null array here; an invalid `sigma` raises in the plugin.
        return self._mu.is_not_null() & self._sigma.is_not_null()

    def sample(self, seed: int | None = None) -> pl.Expr:
        """Draw one LogNormal sample per row, returning a ``Float64`` column.

        Output length follows the surrounding context (frame length under ``select`` / ``with_columns``, partition
        length under ``over`` / ``group_by``). Each row's draw is derived from a per-row sub-seed mixed from ``seed``
        and the row's position, so the result is independent of Polars chunking and thread scheduling.

        Rows with an invalid ``sigma`` raise; rows with a null parameter yield null.

        The output column is named ``"sample"`` when both parameters are constants; column-valued
        parameters keep polars root-name semantics (the name follows the first parameter expression).
        """
        if self._scalar_kwargs is not None:
            return register_plugin(
                "lognormal_sample_scalar", (ROW_INDEX_EXPR,), kwargs={"seed": seed, **self._scalar_kwargs}
            ).alias("sample")
        return register_plugin("lognormal_sample", (self._mu, self._sigma, ROW_INDEX_EXPR), kwargs={"seed": seed})

    def _pdf(self, value: pl.Expr) -> pl.Expr:
        """Density via native ``statrs`` ``Continuous::pdf`` (``0`` for ``value <= 0``)."""
        return self._value_plugin("lognormal_pdf", value)

    def _log_pdf(self, value: pl.Expr) -> pl.Expr:
        """Log-density via native ``Continuous::ln_pdf`` (more accurate than ``pdf().log()``)."""
        return self._value_plugin("lognormal_ln_pdf", value)

    def _cdf(self, value: pl.Expr) -> pl.Expr:
        """Cumulative distribution via native ``ContinuousCDF::cdf`` (``0`` for ``value <= 0``)."""
        return self._value_plugin("lognormal_cdf", value)

    def _sf(self, value: pl.Expr) -> pl.Expr:
        """Survival function via native ``ContinuousCDF::sf`` (accurate in the upper tail).

        ``log_sf`` and ``isf`` inherit the base-class defaults, which compose this native ``sf`` and
        ``ppf`` rather than the generic ``1 - cdf`` fallback.
        """
        return self._value_plugin("lognormal_sf", value)

    def _ppf(self, quantile: pl.Expr) -> pl.Expr:
        """Inverse cdf via the closed-form ``ContinuousCDF::inverse_cdf``.

        A quantile outside ``[0, 1]`` yields null; the endpoints map to the support boundaries
        (``ppf(0) = 0``, ``ppf(1) = +inf``), matching scipy.
        """
        return self._value_plugin("lognormal_ppf", quantile)

    def _moment(self, value: pl.Expr) -> pl.Expr:
        """Gate a closed-form moment on a non-null, valid ``(mu, sigma)`` parameterisation.

        Evaluating ``_checked_sigma`` validates ``sigma`` in Rust (raising on ``sigma <= 0`` or a non-finite parameter)
        and is null when ``sigma`` is null; the ``mu`` term propagates a null ``mu``. So every moment nulls on either
        null input and raises identically on an invalid parameterisation, matching ``Normal._moment``.
        Validation lives in the gate, so ``value`` can read the raw ``self._mu`` / ``self._sigma`` without
        re-validating.
        """
        return pl.when(self._checked_sigma.is_not_null() & self._mu.is_not_null()).then(value)

    def mean(self) -> pl.Expr:
        """Expected value, ``exp(mu + sigma ** 2 / 2)``."""
        return self._moment((self._mu + self._sigma**2 / 2).exp())

    def variance(self) -> pl.Expr:
        """Variance, ``(exp(sigma ** 2) - 1) * exp(2 * mu + sigma ** 2)``."""
        return self._moment(((self._sigma**2).exp() - 1) * (2 * self._mu + self._sigma**2).exp())

    def median(self) -> pl.Expr:
        """Median, ``exp(mu)``."""
        return self._moment(self._mu.exp())

    def entropy(self) -> pl.Expr:
        """Differential entropy, ``mu + 0.5 * log(2 * pi * e * sigma ** 2)``."""
        return self._moment(self._mu + 0.5 * (_TWO_PI_E * self._sigma**2).log())
