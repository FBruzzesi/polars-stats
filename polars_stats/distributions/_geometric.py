from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar

import polars as pl

from polars_stats.distributions._base import (
    DiscreteDistribution,
    coerce_param,
    scalar_float,
    scalar_kwargs,
)

if TYPE_CHECKING:
    from polars_stats._typing import IntoExprColumn

_LN_2 = math.log(2.0)
"""`log(2)`, the constant in the `expm1` identity `_cdf` and `_log_cdf` are spelled through."""

_CDF_DIRECT_COMPLEMENT_MAX = -20.0
"""Crossover of `Geometric._cdf`, in units of `log(sf)`.

Below it `exp(log_tail)` is small enough that the direct `1 - exp(log_tail)` rounds no worse than the
`sinh` identity and cannot round above `1`. The identity itself only breaks where `log_tail / 2`
overflows `sinh`, past `log_tail ~ -1455`, far inside this branch.
"""

_LOG_CDF_LOG1P_MAX = -1.0
"""Crossover of `Geometric._log_cdf`, in units of `log(sf)`.

Below it the cdf sits within `0.63` of `1` and `log1p(-exp(log_tail))` carries that difference
exactly, down through the `cdf ~ 1 - 1e-16` zone where `cdf().log()` reads the tail mass as `0`.
"""


def _smallest_support_point(log_target: pl.Expr, log_failure: pl.Expr) -> pl.Expr:
    """Smallest ``k >= 1`` with ``k * log_failure <= log_target``, the shape both inverses share.

    ``ppf`` and ``isf`` differ only in what they take the log of (``1 - q`` against ``q``); the
    ceiling, the one-ulp correction and the clip to the support floor are common. Every answer is a
    positive integer, so the ceiling is clipped up to ``1.0``, which is also where ``-0.0`` lands at
    the degenerate endpoint of either inverse. A ratio sitting an ulp above an exact integer
    overshoots by one support point, so the candidate steps back down whenever ``k - 1`` already
    satisfies the inequality, compared in the same log domain both sides entered through.
    """
    candidate = (log_target / log_failure).ceil()
    stepped_down = pl.when((candidate - 1.0) * log_failure <= log_target).then(candidate - 1.0).otherwise(candidate)
    return stepped_down.clip(1.0)


class Geometric(DiscreteDistribution):
    """Geometric distribution: the number of trials up to and including the first success.

    Equivalent to ``scipy.stats.geom(p)``. The support is the positive integers: a draw of ``k``
    means trials ``1 .. k - 1`` failed and trial ``k`` succeeded, so ``pmf(1) = p`` and the mass
    decays geometrically above it. Textbooks that count *failures before* the first success start the
    support at ``0`` instead; that is a different parameterisation and not what this class computes.

    Arguments:
        p: Success probability of each trial, with ``0 < p <= 1``. Either a Python ``float`` or an
            ``IntoExprColumn`` (``pl.Expr``, ``pl.Series`` or column name ``str``) carrying one
            probability per row.

    An invalid ``p`` (``p <= 0``, ``p > 1`` or ``NaN``) is not checked at construction; matching every
    other distribution, it raises ``InvalidOperation`` (a ``ComputeError``) when any method is
    evaluated. A null ``p`` propagates to null wherever the result depends on ``p``; the off-support
    constants (``pmf(0) = 0``, ``sf(0) = 1``) do not, as in ``Bernoulli``. Samples are ``UInt64``
    trial counts, so unlike ``Bernoulli`` the degenerate ``p = 0`` point mass is not representable.
    """

    _p: pl.Expr
    _plugin_prefix: ClassVar[str] = "geometric"

    def __init__(self, p: float | IntoExprColumn) -> None:
        self._p = coerce_param(p, name="p")
        self._scalar_kwargs = scalar_kwargs(p=scalar_float(p))

    @property
    def _param_exprs(self) -> tuple[pl.Expr, ...]:
        return (self._p,)

    @property
    def _checked_p(self) -> pl.Expr:
        """``p`` validated in Rust to lie in ``(0, 1]`` (raises otherwise), as a length-n column.

        Validated in Rust so the closed-form methods (moments and pmf/cdf/ppf) report an invalid ``p`` consistently
        with ``sample`` rather than silently computing with a non-positive probability. Null propagates.

        `_checked` validates once for a scalar ``p`` (length-1 input) and per-row for a column,
        returning the raw ``p`` behind the validity gate either way (length-n on both paths).
        """
        return self._checked("geometric_p", self._p)

    def _pmf(self, value: pl.Expr) -> pl.Expr:
        """``(1 - p)**(k - 1) * p`` on the positive integers, ``0`` elsewhere."""
        p = self._checked_p
        k = value.floor()
        is_support = (value >= 1) & (k == value)
        return (
            pl.when(is_support)
            # `k = 1` short-circuits the power term, which is `0 * -inf = nan` at `p = 1`.
            .then(pl.when(k == 1).then(p).otherwise(p * ((k - 1) * (-p).log1p()).exp()))
            .otherwise(0.0)
        )

    def _log_pmf(self, value: pl.Expr) -> pl.Expr:
        """``(k - 1) * log1p(-p) + log(p)`` on the positive integers, ``-inf`` elsewhere.

        Overrides the base ``pmf().log()``, whose ``log(1 - p)`` collapses to ``0.0`` below
        ``p ~ 1.1e-16``. See docs/contributing.md, "Numerical stability". ``k = 1`` reads ``log(p)``
        directly, because ``(k - 1) * log1p(-p)`` is ``0 * -inf = nan`` at ``p = 1``.
        """
        p = self._checked_p
        k = value.floor()
        is_support = (value >= 1) & (k == value)
        return (
            pl.when(is_support)
            .then(pl.when(k == 1).then(p.log()).otherwise((k - 1) * (-p).log1p() + p.log()))
            .otherwise(float("-inf"))
        )

    def _cdf(self, value: pl.Expr) -> pl.Expr:
        """``0`` below the support, else ``-expm1(k * log1p(-p))`` with ``k = floor(value)``.

        Entering through ``log1p(-p)`` keeps a small ``p`` exact, where the literal
        ``1 - (1 - p)**k`` inherits the rounding of ``1 - p`` and of the subtraction against ``1``:
        a few parts in ``1e9`` at ``p = 1e-8``. Polars has no ``expm1``, so it is spelled through the
        identity ``expm1(t) = 2 exp(t / 2) sinh(t / 2)``, and read directly as ``1 - exp(t)`` past
        `_CDF_DIRECT_COMPLEMENT_MAX`.
        """
        p = self._checked_p
        log_tail = value.floor() * (-p).log1p()
        half = log_tail / 2
        return (
            pl.when(value < 1)
            .then(0.0)
            .when(log_tail <= _CDF_DIRECT_COMPLEMENT_MAX)
            .then(1.0 - log_tail.exp())
            .otherwise(-2.0 * half.exp() * half.sinh())
        )

    def _log_cdf(self, value: pl.Expr) -> pl.Expr:
        """``-inf`` below the support, else ``log(1 - exp(k * log1p(-p)))`` with ``k = floor(value)``.

        Two regimes, split at `_LOG_CDF_LOG1P_MAX` where the answer's own magnitude stops dwarfing
        the absolute granularity of the pieces. Below it ``log1p`` carries the difference from ``1``
        exactly; above it the `_cdf` pieces assemble on the log scale, each one large precisely where
        the answer is large, so their rounding stays relative.
        """
        p = self._checked_p
        log_tail = value.floor() * (-p).log1p()
        half = log_tail / 2
        return (
            pl.when(value < 1)
            .then(float("-inf"))
            .when(log_tail <= _LOG_CDF_LOG1P_MAX)
            .then((-log_tail.exp()).log1p())
            .otherwise(_LN_2 + half + (-half.sinh()).log())
        )

    def _sf(self, value: pl.Expr) -> pl.Expr:
        """``1`` below the support, else ``exp(floor(value) * log1p(-p))``.

        Overrides the base ``1 - cdf``, which recomputes ``p`` as ``1 - (1 - p)`` and so quantises it
        to the ``1.1e-16`` spacing of ``1.0``, reaching ``0.0`` below that. ``log1p(-p)`` also avoids
        the rounding of the literal ``1 - p``, a few parts in ``1e9`` at ``p = 1e-8``.
        """
        p = self._checked_p
        return pl.when(value < 1).then(1.0).otherwise((value.floor() * (-p).log1p()).exp())

    def _log_sf(self, value: pl.Expr) -> pl.Expr:
        """``0`` below the support, else ``floor(value) * log1p(-p)``."""
        p = self._checked_p
        return pl.when(value < 1).then(0.0).otherwise(value.floor() * (-p).log1p())

    def _ppf(self, quantile: pl.Expr) -> pl.Expr:
        """Smallest ``k`` with ``cdf(k) >= quantile``: ``ceil(log1p(-q) / log1p(-p))``, clipped to the support floor.

        Null for ``q`` outside ``[0, 1]``. Both sides go through ``log1p``: the literal
        ``log(1 - q)`` collapses to ``0.0`` below ``q ~ 1.1e-16`` and answers ``0`` where the smallest
        support point ``1`` is the only correct answer, and the denominator is the reason in
        `_log_pmf`. The negation is written ``0.0 - q`` so an unsigned quantile column promotes to
        ``Float64`` (polars rejects ``neg`` on an unsigned dtype).

        ``q = 1`` runs the ratio off to ``+inf``, the right answer for an unbounded support.
        ``p = 1`` short-circuits before `_smallest_support_point`, where the ratio would be
        ``-inf / -inf = nan`` and the whole mass sits on ``k = 1``.
        """
        p = self._checked_p
        support_point = _smallest_support_point((0.0 - quantile).log1p(), (-p).log1p())
        return (
            pl.when(quantile.is_between(0, 1))
            .then(pl.when(p == 1).then(1.0).otherwise(support_point))
            .otherwise(pl.lit(None, dtype=pl.Float64()))
        )

    def _isf(self, quantile: pl.Expr) -> pl.Expr:
        """Smallest ``k`` with ``sf(k) <= quantile``: ``ceil(log(q) / log1p(-p))``, clipped to the support floor.

        The exact inverse of `_sf`, entered against ``quantile`` itself rather than through the base
        ``ppf(1 - quantile)``, whose complement throws the tail mass away before the inverse runs.
        Here ``q = 1`` degenerates to ``-0.0`` and ``q = 0`` flows through as ``+inf``. At ``p = 1``
        every survival quantile inverts to ``k = 1``, including ``q = 0``, where ``sf(1) = 0``
        already satisfies the inequality.
        """
        p = self._checked_p
        support_point = _smallest_support_point(quantile.log(), (-p).log1p())
        return (
            pl.when(quantile.is_between(0, 1))
            .then(pl.when(p == 1).then(1.0).otherwise(support_point))
            .otherwise(pl.lit(None, dtype=pl.Float64()))
        )

    def mean(self) -> pl.Expr:
        """Expected value, ``1 / p``."""
        return 1 / self._checked_p

    def variance(self) -> pl.Expr:
        """Variance, ``(1 - p) / p**2``."""
        p = self._checked_p
        return (1 - p) / p**2

    def std(self) -> pl.Expr:
        """Standard deviation, ``sqrt(1 - p) / p``.

        Overrides the base-class ``variance().sqrt()``, which squares ``p`` and then unsquares
        it: the round trip overflows about 150 decades before ``sqrt(1 - p) / p`` does.
        """
        p = self._checked_p
        return (1 - p).sqrt() / p

    def entropy(self) -> pl.Expr:
        """Shannon entropy, ``(-(1 - p) * log1p(-p) - p * log(p)) / p``, with the ``0`` limit at ``p = 1``.

        Uses the convention ``0 * log 0 = 0`` so the degenerate ``p = 1`` point mass has entropy ``0``.
        The first term goes through ``log1p`` for the reason in `_log_pmf`.
        """
        p = self._checked_p
        q = 1 - p

        return pl.when(p == 1).then(0.0).otherwise((-q * (-p).log1p() - p * p.log()) / p)
