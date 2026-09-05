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


class Exponential(ContinuousDistribution):
    """Exponential distribution with rate ``rate`` (λ).

    Equivalent to ``scipy.stats.expon(scale=1 / rate)``. The API exposes ``rate`` (the ``statrs``
    parameterisation) rather than scipy's ``scale = 1 / rate``: it is the natural parameter and
    avoids the divide-by-zero footgun of passing ``scale=0``.

    Arguments:
        rate: Rate parameter λ, with ``rate > 0``. Either a Python ``float`` or an ``IntoExprColumn``
            (``pl.Expr``, ``pl.Series`` or column name ``str``) carrying one rate per row.

    An invalid ``rate`` (``rate <= 0`` or ``NaN``) is not checked at construction; matching every
    other distribution, it raises ``InvalidOperation`` (a ``ComputeError``) when any method is
    evaluated. The support is ``x >= 0``: ``pdf`` and ``cdf`` are ``0`` for ``x < 0``, and ``sf`` is
    ``1`` there.

    A null ``rate`` propagates to null wherever the result depends on it; the below-support constants
    (``pdf`` and ``cdf`` ``0``, ``sf`` ``1``, ``log_pdf`` and ``log_cdf`` ``-inf``, ``log_sf`` ``0``)
    do not. Both inverses null at every quantile, in range and out.

    The value-keyed methods compute in Rust, so an invalid ``rate`` is reported whichever branch the
    value selects. The moments stay in Polars, reading ``rate`` through the same Rust validator.
    """

    _rate: pl.Expr
    _plugin_prefix: ClassVar[str] = "exponential"

    def __init__(self, rate: float | IntoExprColumn) -> None:
        self._rate = coerce_param(rate, name="rate")
        self._scalar_kwargs = scalar_kwargs(rate=scalar_float(rate))

    @property
    def _param_exprs(self) -> tuple[pl.Expr, ...]:
        return (self._rate,)

    @property
    def _checked_rate(self) -> pl.Expr:
        """``rate`` (λ) validated in Rust to be strictly positive (raises otherwise), as a length-n column.

        Null propagates. Read only by the moments, so they report an invalid ``rate`` consistently
        with ``sample`` rather than silently computing with a non-positive one; the value-keyed
        methods validate inside their own plugin instead.
        """
        return self._checked("exponential_rate", self._rate)

    def _pdf(self, value: pl.Expr) -> pl.Expr:
        """``rate * exp(-rate * x)`` on ``x >= 0``, ``0`` for ``x < 0``."""
        return self._value_plugin("exponential_pdf", value)

    def _log_pdf(self, value: pl.Expr) -> pl.Expr:
        """``log(rate) - rate * x`` on ``x >= 0``, ``-inf`` for ``x < 0``."""
        return self._value_plugin("exponential_ln_pdf", value)

    def _cdf(self, value: pl.Expr) -> pl.Expr:
        """``1 - exp(-rate * x)`` on ``x >= 0``, ``0`` for ``x < 0``."""
        return self._value_plugin("exponential_cdf", value)

    def _log_cdf(self, value: pl.Expr) -> pl.Expr:
        """``log(cdf)`` in the left tail, ``log1p(-sf)`` in the right; ``-inf`` for ``x <= 0``."""
        return self._value_plugin("exponential_ln_cdf", value)

    def _sf(self, value: pl.Expr) -> pl.Expr:
        """``exp(-rate * x)`` on ``x >= 0``, ``1`` for ``x < 0``."""
        return self._value_plugin("exponential_sf", value)

    def _log_sf(self, value: pl.Expr) -> pl.Expr:
        """``-rate * x`` on ``x >= 0``, ``0`` for ``x < 0``."""
        return self._value_plugin("exponential_ln_sf", value)

    def _ppf(self, quantile: pl.Expr) -> pl.Expr:
        """``-log1p(-q) / rate``; null for ``q`` outside ``[0, 1]``."""
        return self._value_plugin("exponential_ppf", quantile)

    def _isf(self, quantile: pl.Expr) -> pl.Expr:
        """``-log(q) / rate``; null for ``q`` outside ``[0, 1]``.

        Overrides the base ``ppf(1 - quantile)``, which forms the complement and then undoes it.
        """
        return self._value_plugin("exponential_isf", quantile)

    def mean(self) -> pl.Expr:
        """Expected value, ``1 / rate``."""
        return 1 / self._checked_rate

    def variance(self) -> pl.Expr:
        """Variance, ``1 / rate**2``."""
        return 1 / self._checked_rate**2

    def std(self) -> pl.Expr:
        """Standard deviation, ``1 / rate``, the same expression as ``mean``.

        Overrides the base-class ``variance().sqrt()``, which squares the rate and then unsquares
        it: the round trip saturates about 300 decades before ``1 / rate`` does.
        """
        return 1 / self._checked_rate

    def median(self) -> pl.Expr:
        """Median, ``log(2) / rate``."""
        return math.log(2) / self._checked_rate

    def entropy(self) -> pl.Expr:
        """Differential entropy, ``1 - log(rate)``."""
        return 1 - self._checked_rate.log()
