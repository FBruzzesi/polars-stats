from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar

import polars as pl

from polars_stats.distributions._base import (
    ContinuousDistribution,
    coerce_param,
    scalar_float,
    scalar_kwargs,
)

if TYPE_CHECKING:
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
    evaluated. A null ``rate`` propagates to null. The support is ``x >= 0``: ``pdf`` and ``cdf`` are
    ``0`` for ``x < 0``, and ``sf`` is ``1`` there.
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

        Validated in Rust so every closed-form method (moments and pdf/cdf/ppf) reports an invalid
        ``rate`` consistently with ``sample`` rather than silently computing with a non-positive rate.
        Null propagates.

        `_checked` validates once for a scalar ``rate`` (length-1 input) and per-row for a column,
        returning the raw ``rate`` behind the validity gate either way (length-n on both paths). The
        analogue of ``Bernoulli._checked_p``: the validated quantity is the parameter itself.
        """
        return self._checked("exponential_rate", self._rate)

    def _pdf(self, value: pl.Expr) -> pl.Expr:
        """``rate * exp(-rate * x)`` on ``x >= 0``, ``0`` for ``x < 0``."""
        r = self._checked_rate
        return pl.when(value >= 0).then(r * (-r * value).exp()).otherwise(0.0)

    def _log_pdf(self, value: pl.Expr) -> pl.Expr:
        """``log(rate) - rate * x`` on ``x >= 0``, ``-inf`` for ``x < 0``."""
        r = self._checked_rate
        return pl.when(value >= 0).then(r.log() - r * value).otherwise(float("-inf"))

    def _cdf(self, value: pl.Expr) -> pl.Expr:
        """``1 - exp(-rate * x)`` on ``x >= 0``, ``0`` for ``x < 0``."""
        r = self._checked_rate
        return pl.when(value >= 0).then(1 - (-r * value).exp()).otherwise(0.0)

    def _log_cdf(self, value: pl.Expr) -> pl.Expr:
        """``log(1 - exp(-rate * x))`` via ``log1p(-sf)``; ``-inf`` for ``x <= 0``.

        Overrides the base ``log(cdf)``, which loses relative precision as ``cdf -> 1`` (the deep
        right tail, where ``log_cdf -> 0``): ``cdf`` rounds to ~1 and its log to a tiny, inaccurate
        value. ``log1p(-sf)`` keeps full precision there, since ``sf = exp(-rate * x)`` is small and
        ``log1p`` is accurate near 0. Polars exposes no ``expm1``, so the ``cdf -> 0`` left tail is no
        more accurate than the base (the cancellation in ``1 - exp(-rate * x)`` is unavoidable in pure
        Polars); that regime keeps full precision in ``cdf`` / ``sf`` instead.
        """
        return (-self._sf(value)).log1p()

    def _sf(self, value: pl.Expr) -> pl.Expr:
        """``exp(-rate * x)`` on ``x >= 0``, ``1`` for ``x < 0`` (closed form, accurate in the upper tail)."""
        r = self._checked_rate
        return pl.when(value >= 0).then((-r * value).exp()).otherwise(1.0)

    def _log_sf(self, value: pl.Expr) -> pl.Expr:
        """``-rate * x`` on ``x >= 0``, ``0`` for ``x < 0``."""
        r = self._checked_rate
        return pl.when(value >= 0).then(-r * value).otherwise(0.0)

    def _ppf(self, quantile: pl.Expr) -> pl.Expr:
        """``-log(1 - q) / rate``; null for ``q`` outside ``[0, 1]``."""
        return (
            pl.when(quantile.is_between(0, 1))
            .then(-(1 - quantile).log() / self._checked_rate)
            .otherwise(pl.lit(None, dtype=pl.Float64()))
        )

    def mean(self) -> pl.Expr:
        """Expected value, ``1 / rate``."""
        return 1 / self._checked_rate

    def variance(self) -> pl.Expr:
        """Variance, ``1 / rate**2``."""
        return 1 / self._checked_rate**2

    def median(self) -> pl.Expr:
        """Median, ``log(2) / rate``."""
        return math.log(2) / self._checked_rate

    def entropy(self) -> pl.Expr:
        """Differential entropy, ``1 - log(rate)``."""
        return 1 - self._checked_rate.log()
