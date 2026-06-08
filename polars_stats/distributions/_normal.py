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
)

if TYPE_CHECKING:
    from polars_stats._typing import IntoExprColumn, PolarsDataType


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
    _sample_dtype: ClassVar[PolarsDataType] = pl.Float64()

    def __init__(
        self,
        mean: float | IntoExprColumn = 0.0,
        std_dev: float | IntoExprColumn = 1.0,
    ) -> None:
        self._mean = coerce_param(mean, name="mean")
        self._std_dev = coerce_param(std_dev, name="std_dev")
        # Constant parameters (if any) enable the fast sampler path; `None` falls back to the per-row plugin.
        self._mean_scalar = scalar_float(mean)
        self._std_dev_scalar = scalar_float(std_dev)

    def _value_plugin(self, function_name: str, value: pl.Expr) -> pl.Expr:
        """Register a value-keyed Rust plugin call ``f(value, mean, std_dev)``.

        Validation of ``std_dev`` happens inside the plugin, so every value-keyed method reports an
        invalid scale consistently; null inputs propagate per row.
        """
        return register_plugin(function_name, (value, self._mean, self._std_dev))

    @property
    def _checked_std_dev(self) -> pl.Expr:
        """``std_dev`` validated in Rust against the full ``(mean, std_dev)`` parameterisation."""
        # Mirrors ``Uniform.range`` / ``Bernoulli._checked_p``: the closed-form moments derive from this
        # single FFI round-trip, so they report an invalid parameterisation (``std_dev <= 0``, or a
        # non-finite parameter) as a ``ComputeError`` consistently with the value-keyed methods. Null in
        # either parameter propagates to null, so a moment built on this nulls when either input is null.
        return register_plugin("normal_std_dev", (self._mean, self._std_dev))

    def _valid_mask(self) -> pl.Expr:
        # Only null parameters are masked to a null array here; an invalid `std_dev` raises in the plugin.
        return self._mean.is_not_null() & self._std_dev.is_not_null()

    def sample(self, seed: int | None = None) -> pl.Expr:
        """Draw one Normal sample per row, returning a ``Float64`` column.

        Output length follows the surrounding context (frame length under ``select`` / ``with_columns``,
        partition length under ``over`` / ``group_by``). Each row's draw is derived from a per-row sub-seed mixed from
        ``seed`` and the row's position, so the result is independent of Polars chunking and thread scheduling.

        Rows with an invalid ``std_dev`` raise; rows with a null parameter yield null.
        """
        if self._mean_scalar is not None and self._std_dev_scalar is not None:
            return register_plugin(
                "normal_sample_scalar",
                (ROW_INDEX_EXPR,),
                kwargs={"seed": seed, "mean": self._mean_scalar, "std_dev": self._std_dev_scalar},
            )
        return register_plugin("normal_sample", (self._mean, self._std_dev, ROW_INDEX_EXPR), kwargs={"seed": seed})

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

    def _moment(self, value: pl.Expr) -> pl.Expr:
        """Gate a closed-form moment on a non-null, valid ``(mean, std_dev)`` parameterisation.

        Evaluating ``_checked_std_dev`` validates ``std_dev`` in Rust (raising on ``std_dev <= 0`` or a
        non-finite parameter) and is null when ``std_dev`` is null; the ``mean`` term propagates a null
        ``mean``. So every moment nulls on either null input and raises identically on an invalid scale,
        regardless of which parameter ``value`` itself references. Validation lives in the gate, so
        ``value`` can read the raw ``self._std_dev`` / ``self._mean`` without re-validating.
        """
        return pl.when(self._checked_std_dev.is_not_null() & self._mean.is_not_null()).then(value)

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
