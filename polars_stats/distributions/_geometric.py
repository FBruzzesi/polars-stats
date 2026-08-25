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

# Past this log-tail depth the cdf complement `exp(x)` is small enough that `1 - exp(x)`
# rounds no worse than the sinh identity, and cannot round above `1`.
_CDF_DIRECT_TAIL = -20.0


class Geometric(DiscreteDistribution):
    """Geometric distribution: the number of trials up to and including the first success.

    Equivalent to ``scipy.stats.geom(p)``. The support is the positive integers: a draw of ``k``
    means trials ``1 .. k - 1`` failed and trial ``k`` succeeded, so ``pmf(1) = p`` and the mass
    decays geometrically above it.

    Arguments:
        p: Success probability of each trial, with ``0 < p <= 1``. Either a Python ``float`` or an
            ``IntoExprColumn`` (``pl.Expr``, ``pl.Series`` or column name ``str``) carrying one
            probability per row.

    An invalid ``p`` (``p <= 0``, ``p > 1`` or ``NaN``) is not checked at construction; matching every
    other distribution, it raises ``InvalidOperation`` (a ``ComputeError``) when any method is
    evaluated. A null ``p`` propagates to null. Samples are ``UInt64`` trial counts, so unlike
    ``Bernoulli`` the degenerate ``p = 0`` point mass is not representable.
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
        is_support = (value >= 1) & (value.floor() == value)
        return (
            pl.when(is_support)
            .then(
                # `k = 1` short-circuits before the power term: `(k - 1) * log1p(-p)` degenerates to
                # `0 * -inf = nan` at `p = 1`, where the mass is exactly `p`.
                pl.when(value <= 1).then(p).otherwise(p * ((value.floor() - 1) * (-p).log1p()).exp())
            )
            .otherwise(0.0)
        )

    def _log_pmf(self, value: pl.Expr) -> pl.Expr:
        """``(k - 1) * log1p(-p) + log(p)`` on the positive integers, ``-inf`` elsewhere.

        Overrides the base ``pmf().log()``, whose ``log(1 - p)`` collapses to ``0.0`` below
        ``p ~ 1.1e-16``. See docs/contributing.md, "Numerical stability". The ``k = 1`` case reads
        ``log(p)`` directly because ``(k - 1) * log1p(-p)`` degenerates to ``0 * -inf = nan`` at
        ``p = 1``, where the answer is exactly ``0``.
        """
        p = self._checked_p
        is_support = (value >= 1) & (value.floor() == value)
        return (
            pl.when(is_support)
            .then(pl.when(value.floor() == 1).then(p.log()).otherwise((value.floor() - 1) * (-p).log1p() + p.log()))
            .otherwise(float("-inf"))
        )

    def _cdf(self, value: pl.Expr) -> pl.Expr:
        """``0`` for ``value < 1``, else ``-expm1(floor(value) * log1p(-p))``.

        The mass sits at ``p``-scale for small ``k``, so the naive ``1 - (1 - p)**floor(value)``
        inherits the rounding of ``1 - p`` and of the subtraction against ``1`` -- a few parts in
        ``1e9`` of relative error at ``p = 1e-8``, where the true answer is ``1e-8``. Entering
        through ``log1p(-p)`` keeps the parameter exact; past ``x = -20`` the complement is read
        directly as ``1 - exp(x)``, which cannot round above ``1``; the exponential itself is
        evaluated through the identity ``expm1(x) = 2 exp(x/2) sinh(x/2)`` (polars has no
        ``expm1``), whose only weak spot is ``x/2`` overflowing ``sinh`` below ``x ~ -1455`` --
        far inside the direct-complement branch.
        """
        p = self._checked_p
        x = value.floor() * (-p).log1p()
        return (
            pl.when(value < 1)
            .then(0.0)
            .when(x <= _CDF_DIRECT_TAIL)
            .then(1.0 - x.exp())
            .otherwise(-2.0 * (x / 2).exp() * (x / 2).sinh())
        )

    def _log_cdf(self, value: pl.Expr) -> pl.Expr:
        """``-inf`` for ``value < 1``, else ``log(1 - exp(floor(value) * log1p(-p)))``.

        Two regimes, split where the answer's own magnitude stops dwarfing the absolute granularity
        of the pieces. Below ``x = -1`` the cdf sits within ``0.63`` of ``1``, and ``log1p(-exp(x))``
        carries that difference exactly -- down through the ``cdf ~ 1 - 1e-16`` saturation zone,
        where ``log(cdf)`` reads the tail mass rounded away against ``1`` and answers ``0``. Above
        it, the answer assembles from the `_cdf` pieces on the log scale, ``log(2 exp(x/2)
        sinh(x/2))``: every piece is huge precisely where the answer is huge, so their rounding
        stays relative.
        """
        p = self._checked_p
        x = value.floor() * (-p).log1p()
        return (
            pl.when(value < 1)
            .then(float("-inf"))
            .when(x <= -1.0)
            .then((-x.exp()).log1p())
            .otherwise(math.log(2.0) + x / 2 + (-(x / 2).sinh()).log())
        )

    def _sf(self, value: pl.Expr) -> pl.Expr:
        """``1`` for ``value < 1``, ``exp(floor(value) * log1p(-p))`` for ``value >= 1``.

        A closed form that stays accurate in the upper tail, unlike the base ``1 - cdf``, which
        recomputes ``p`` as ``1 - (1 - p)`` and so quantises it to the ``1.1e-16`` spacing of ``1.0``,
        reaching ``0.0`` below that. Entering through ``log1p(-p)`` also avoids the rounding of the
        literal ``1 - p``, visible as a few parts in ``1e9`` of relative error at ``p = 1e-8``.
        """
        p = self._checked_p
        return pl.when(value < 1).then(1.0).otherwise((value.floor() * (-p).log1p()).exp())

    def _log_sf(self, value: pl.Expr) -> pl.Expr:
        """``0`` for ``value < 1``, ``floor(value) * log1p(-p)`` for ``value >= 1``."""
        p = self._checked_p
        return pl.when(value < 1).then(0.0).otherwise(value.floor() * (-p).log1p())

    def _ppf(self, quantile: pl.Expr) -> pl.Expr:
        """Smallest ``k`` with ``cdf(k) >= quantile``: ``ceil(log1p(-q) / log1p(-p))``, clipped to the support floor.

        Null for ``q`` outside ``[0, 1]``. Both sides go through ``log1p``: the naive
        ``log(1 - q) / log(1 - p)`` collapses its numerator to ``0.0`` below ``q ~ 1.1e-16`` and
        answers ``0`` where the smallest support point ``1`` is the only correct answer, while the
        denominator side is the reason in ``_log_pmf``. The negation is written ``0.0 - q`` so an
        unsigned quantile column promotes to ``Float64`` (polars rejects ``neg`` on an unsigned
        dtype).

        Every answer lives on the positive integers, so the ``ceil`` is clipped up to ``1.0``: at
        ``q = 0`` the formula degenerates to ``-0.0``, and at ``p < 1, q = 1`` the ratio runs off to
        ``+inf``. At ``p = 1`` the ratio is ``-inf / -inf = nan``, so that parameter degeneracy
        short-circuits first: the whole mass sits on ``k = 1``.

        The raw ceiling can overshoot by one support point when the ratio lands an ulp above an
        exact integer -- the stored quantile then sits a hair below the cdf step it belongs to --
        so the candidate is stepped back down whenever ``k - 1`` already satisfies its own defining
        inequality, compared in the log domain both sides entered through.

        The result stays a ``Float64`` support point rather than an integral dtype, matching the other
        distributions' inverses and leaving the wrapper's ``NaN -> NaN`` contract representable.
        """
        p = self._checked_p
        tail_log = (-p).log1p()
        target_log = (0.0 - quantile).log1p()
        candidate = (target_log / tail_log).ceil()
        stepped = (
            pl.when((candidate - 1.0) * tail_log <= target_log).then(candidate - 1.0).otherwise(candidate).clip(1.0)
        )
        return (
            pl.when(quantile.is_between(0, 1))
            .then(pl.when(p == 1).then(1.0).otherwise(stepped))
            .otherwise(pl.lit(None, dtype=pl.Float64()))
        )

    def _isf(self, quantile: pl.Expr) -> pl.Expr:
        """Smallest ``k`` with ``sf(k) <= quantile``: ``ceil(log(q) / log1p(-p))``, clipped to the support floor.

        The exact inverse of `_sf`, entered against ``quantile`` itself rather than through the base
        ``ppf(1 - quantile)``, whose complement throws the tail mass away before the inverse runs.
        As in `_ppf`, every answer lives on the positive integers, so the ``ceil`` is clipped up to
        ``1.0``: at ``q = 1`` the formula degenerates to ``-0.0``, and ``q = 0`` flows through it as
        ``+inf``. At ``p = 1`` the ratio is ``nan`` or ``-inf / -inf``, so that parameter degeneracy
        short-circuits first: every survival quantile inverts to the single mass point ``k = 1``,
        including ``q = 0``, where ``sf(1) = 0`` already satisfies the inequality.

        The same one-ulp overshoot as in `_ppf` applies -- an exact-integer ratio landing an ulp high
        skips a support point -- so the candidate steps back down whenever ``k - 1`` already
        satisfies ``sf(k - 1) <= q``, compared as ``(k - 1) * log1p(-p) <= log(q)``.
        """
        p = self._checked_p
        tail_log = (-p).log1p()
        target_log = quantile.log()
        candidate = (target_log / tail_log).ceil()
        stepped = (
            pl.when((candidate - 1.0) * tail_log <= target_log).then(candidate - 1.0).otherwise(candidate).clip(1.0)
        )
        return (
            pl.when(quantile.is_between(0, 1))
            .then(pl.when(p == 1).then(1.0).otherwise(stepped))
            .otherwise(pl.lit(None, dtype=pl.Float64()))
        )

    def mean(self) -> pl.Expr:
        """Expected value, ``1 / p``."""
        return 1 / self._checked_p

    def variance(self) -> pl.Expr:
        """Variance, ``(1 - p) / p**2``."""
        return (1 - self._checked_p) / self._checked_p**2

    def std(self) -> pl.Expr:
        """Standard deviation, ``sqrt(1 - p) / p``.

        Overrides the base-class ``variance().sqrt()``, which squares ``p`` and then unsquares
        it: the round trip overflows about 150 decades before ``sqrt(1 - p) / p`` does.
        """
        return (1 - self._checked_p).sqrt() / self._checked_p

    def entropy(self) -> pl.Expr:
        """Shannon entropy, ``(-(1 - p) * log1p(-p) - p * log(p)) / p``, with the ``0`` limit at ``p = 1``.

        Uses the convention ``0 * log 0 = 0`` so the degenerate ``p = 1`` point mass has entropy ``0``.
        The first term goes through ``log1p`` for the reason in ``_log_pmf``.
        """
        p = self._checked_p
        q = 1 - p

        return pl.when(p == 1).then(0.0).otherwise((-q * (-p).log1p() - p * p.log()) / p)
