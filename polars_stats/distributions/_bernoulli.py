from __future__ import annotations

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


class Bernoulli(DiscreteDistribution):
    """Bernoulli distribution with success probability ``p``.

    Equivalent to ``scipy.stats.bernoulli(p)``.

    Arguments:
        p: Success probability, with ``0 <= p <= 1``. Either a Python ``float`` or an ``IntoExprColumn``
            (``pl.Expr``, ``pl.Series`` or column name ``str``) carrying one probability per row.

    An invalid ``p`` (``p < 0``, ``p > 1`` or ``NaN``) is not checked at construction; matching every other
    distribution, it raises ``InvalidOperation`` (a ``ComputeError``) when any method is evaluated.

    A null ``p`` propagates to null wherever the result depends on ``p``; the off-support constants
    (``pmf(2) = 0``, ``cdf(-1) = 0``, ``sf(1) = 0``) do not.

    The value-keyed methods compute in Rust, so an invalid ``p`` is reported whichever branch the
    value selects. The moments stay in Polars, reading ``p`` through the same Rust validator.
    """

    _p: pl.Expr
    _plugin_prefix: ClassVar[str] = "bernoulli"

    def __init__(self, p: float | IntoExprColumn) -> None:
        self._p = coerce_param(p, name="p")
        self._scalar_kwargs = scalar_kwargs(p=scalar_float(p))

    @property
    def _param_exprs(self) -> tuple[pl.Expr, ...]:
        return (self._p,)

    @property
    def _checked_p(self) -> pl.Expr:
        """``p`` validated in Rust to lie in ``[0, 1]`` (raises otherwise), as a length-n column. Null propagates.

        Read only by the moments; the value-keyed methods validate inside their own plugin instead, which is what
        makes them immune to a branch never reaching the validator.
        """
        return self._checked("bernoulli_proba", self._p)

    def _pmf(self, value: pl.Expr) -> pl.Expr:
        """``1 - p`` at 0, ``p`` at 1, ``0`` elsewhere; the off-support ``0`` carries no ``p``."""
        return self._value_plugin("bernoulli_pmf", value)

    def _log_pmf(self, value: pl.Expr) -> pl.Expr:
        """``log1p(-p)`` at 0, ``log(p)`` at 1, ``-inf`` elsewhere.

        ``log1p``, not ``log(1 - p)``: the latter collapses to ``0.0`` below ``p ~ 1.1e-16``.
        """
        return self._value_plugin("bernoulli_ln_pmf", value)

    def _cdf(self, value: pl.Expr) -> pl.Expr:
        """``0`` for ``value < 0``, ``1 - p`` for ``0 <= value < 1``, ``1`` for ``value >= 1``."""
        return self._value_plugin("bernoulli_cdf", value)

    def _log_cdf(self, value: pl.Expr) -> pl.Expr:
        """``-inf`` for ``value < 0``, ``log1p(-p)`` for ``0 <= value < 1``, ``0`` for ``value >= 1``."""
        return self._value_plugin("bernoulli_ln_cdf", value)

    def _sf(self, value: pl.Expr) -> pl.Expr:
        """``1`` for ``value < 0``, ``p`` for ``0 <= value < 1``, ``0`` for ``value >= 1``."""
        return self._value_plugin("bernoulli_sf", value)

    def _log_sf(self, value: pl.Expr) -> pl.Expr:
        """``0`` for ``value < 0``, ``log(p)`` for ``0 <= value < 1``, ``-inf`` for ``value >= 1``."""
        return self._value_plugin("bernoulli_ln_sf", value)

    def _ppf(self, quantile: pl.Expr) -> pl.Expr:
        """Smallest ``x`` with ``cdf(x) >= quantile``: ``0.0`` if ``quantile <= 1 - p`` else ``1.0``.

        A quantile outside ``[0, 1]`` yields null. The support point comes back as ``Float64``, as scipy reports it.
        """
        return self._value_plugin("bernoulli_ppf", quantile)

    def _isf(self, quantile: pl.Expr) -> pl.Expr:
        """Smallest ``x`` with ``sf(x) <= quantile``: ``1.0`` if ``p > quantile`` else ``0.0``.

        A quantile outside ``[0, 1]`` yields null.
        """
        return self._value_plugin("bernoulli_isf", quantile)

    def mean(self) -> pl.Expr:
        """Expected value, ``p``."""
        return self._checked_p

    def variance(self) -> pl.Expr:
        """Variance, ``p * (1 - p)``."""
        p = self._checked_p
        return p * (1 - p)

    def entropy(self) -> pl.Expr:
        """Shannon entropy, ``-p * log(p) - (1 - p) * log1p(-p)``.

        Uses the convention ``0 * log 0 = 0`` so the result is ``0`` at the degenerate endpoints ``p in {0, 1}``.
        The second term goes through ``log1p`` for the reason in ``_log_pmf``: ``log(1 - p)`` collapses to ``0.0``
        below ``p ~ 1.1e-16``.
        """
        p = self._checked_p
        q = 1 - p

        return pl.when((p == 0) | (p == 1)).then(0.0).otherwise(-p * p.log() - q * (-p).log1p())
