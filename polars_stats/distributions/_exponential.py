from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar

import polars as pl

from polars_stats.distributions._base import (
    ContinuousDistribution,
    coerce_param,
    expm1,
    scalar_float,
    scalar_kwargs,
)

if TYPE_CHECKING:
    from polars_stats._typing import IntoExprColumn

_CDF_SINH_MAX = 1.0
"""Crossover of `Exponential._cdf`, in units of `rate * x`.

Above `rate * x = 1` the plain `1 - exp(-t)` is already exact, and the `sinh` identity that replaces
it below would overflow past `t ~ 1420`. The two branches agree to `1e-16` here.
"""


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
        """``rate * exp(-rate * x)`` on ``x >= 0``, ``0`` for ``x < 0``, keeping the subnormal range exact.

        Reassociated as ``(rate * exp(-rate * x / 2)) * exp(-rate * x / 2)``: the same product with
        the rounding moved to the end. Written literally, ``exp(-rate * x)`` rounds into the
        gradual-underflow range while the scale is still to be applied, and the multiply then
        magnifies what the subnormal threw away. Halving the exponent keeps the intermediate normal,
        so only the final multiply underflows, at the cost of one multiply and no branch.
        """
        r = self._checked_rate
        half = (-r * value / 2).exp()
        return pl.when(value >= 0).then((r * half) * half).otherwise(0.0)

    def _log_pdf(self, value: pl.Expr) -> pl.Expr:
        """``log(rate) - rate * x`` on ``x >= 0``, ``-inf`` for ``x < 0``."""
        r = self._checked_rate
        return pl.when(value >= 0).then(r.log() - r * value).otherwise(float("-inf"))

    def _cdf(self, value: pl.Expr) -> pl.Expr:
        """``1 - exp(-rate * x)`` on ``x >= 0``, ``0`` for ``x < 0``, keeping the left tail exact.

        ``1 - exp(-t)`` cancels to ``0`` below ``t ~ 1.1e-16``, so the small branch reads it as
        ``-expm1(-t)``. ``sinh`` overflows above ``t ~ 1420``, hence the split at `_CDF_SINH_MAX`,
        where the plain form is already exact; the two agree to ``1e-16`` at the crossover.
        """
        t = self._checked_rate * value
        return pl.when(value < 0).then(0.0).when(t < _CDF_SINH_MAX).then(-expm1(-t)).otherwise(1 - (-t).exp())

    def _log_cdf(self, value: pl.Expr) -> pl.Expr:
        """``log(cdf)`` in the left tail, ``log1p(-sf)`` in the right; ``-inf`` for ``x <= 0``.

        As ``cdf -> 1`` the cdf rounds to ~1 and its log to a tiny, inaccurate value, so that side
        goes through ``log1p`` of the small ``sf``. As ``cdf -> 0`` the log is well conditioned and
        ``_cdf`` is exact there, so that side is simply its log.
        """
        return (
            pl.when(self._checked_rate * value < 1).then(self._cdf(value).log()).otherwise((-self._sf(value)).log1p())
        )

    def _sf(self, value: pl.Expr) -> pl.Expr:
        """``exp(-rate * x)`` on ``x >= 0``, ``1`` for ``x < 0`` (closed form, accurate in the upper tail)."""
        r = self._checked_rate
        return pl.when(value >= 0).then((-r * value).exp()).otherwise(1.0)

    def _log_sf(self, value: pl.Expr) -> pl.Expr:
        """``-rate * x`` on ``x >= 0``, ``0`` for ``x < 0``."""
        r = self._checked_rate
        return pl.when(value >= 0).then(-r * value).otherwise(0.0)

    def _ppf(self, quantile: pl.Expr) -> pl.Expr:
        """``-log1p(-q) / rate``; null for ``q`` outside ``[0, 1]``.

        Through ``log1p`` rather than ``log(1 - q)``: the latter rounds ``1 - q`` to exactly ``1``
        below ``q ~ 1.1e-16`` and collapses to ``-0.0``. The negation is written ``0.0 - q`` so an
        unsigned quantile column promotes to ``Float64`` (polars rejects ``neg`` on an unsigned
        dtype).
        """
        return (
            pl.when(quantile.is_between(0, 1))
            .then(-(0.0 - quantile).log1p() / self._checked_rate)
            .otherwise(pl.lit(None, dtype=pl.Float64()))
        )

    def _isf(self, quantile: pl.Expr) -> pl.Expr:
        """``-log(q) / rate``, the exact inverse survival function.

        Overrides the base ``ppf(1 - quantile)``, which forms the complement and then undoes it.
        The closed form never builds it, and is exact for every ``q`` in ``(0, 1]``.
        """
        return (
            pl.when(quantile.is_between(0, 1))
            .then(-quantile.log() / self._checked_rate)
            .otherwise(pl.lit(None, dtype=pl.Float64()))
        )

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
