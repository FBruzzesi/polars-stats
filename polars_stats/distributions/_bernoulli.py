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

    Arguments:
        p: Success probability of Bernoulli distribution. Either a Python ``float`` or an ``IntoExprColumn``
            (``pl.Expr``, ``pl.Series`` or column name ``str``) carrying one probability per row.
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
        """``p`` validated in Rust to lie in ``[0, 1]`` (raises otherwise), as a length-n column.

        Validated in Rust so the closed-form methods (moments and pmf/cdf/ppf) report an invalid ``p`` consistently
        with ``sample`` rather than silently computing a negative probability. Null propagates.

        `_checked` validates once for a scalar ``p`` (length-1 input) and per-row for a column,
        returning the raw ``p`` behind the validity gate either way (length-n on both paths).
        """
        return self._checked("bernoulli_proba", self._p)

    def _pmf(self, value: pl.Expr) -> pl.Expr:
        """``1 - p`` at 0, ``p`` at 1, ``0`` elsewhere."""
        p = self._checked_p
        return pl.when(value.eq(0)).then(1 - p).when(value.eq(1)).then(p).otherwise(0.0)

    def _cdf(self, value: pl.Expr) -> pl.Expr:
        """``0`` for ``value < 0``, ``1 - p`` for ``0 <= value < 1``, ``1`` for ``value >= 1``."""
        return pl.when(value.lt(0)).then(0.0).when(value.lt(1)).then(1 - self._checked_p).otherwise(1.0)

    def _ppf(self, quantile: pl.Expr) -> pl.Expr:
        """Smallest ``x`` with ``cdf(x) >= quantile``: ``0.0`` if ``quantile <= 1 - p`` else ``1.0``.

        The comparison's ``Boolean`` is cast to ``Float64`` so ``ppf`` / ``isf`` / ``median`` return the
        support point numerically, matching scipy and every other distribution and leaving the wrapper's
        ``NaN -> NaN`` contract representable (``Boolean`` has no ``NaN``).
        """
        return (quantile > 1 - self._checked_p).cast(pl.Float64())

    def mean(self) -> pl.Expr:
        """Expected value, ``p``."""
        return self._checked_p

    def variance(self) -> pl.Expr:
        """Variance, ``p * (1 - p)``."""
        p = self._checked_p
        return p * (1 - p)

    def entropy(self) -> pl.Expr:
        """Shannon entropy, ``-p * log(p) - (1 - p) * log(1 - p)``.

        Uses the convention ``0 * log 0 = 0`` so the result is ``0`` at the degenerate endpoints ``p in {0, 1}``.
        """
        p = self._checked_p
        q = 1 - p

        return pl.when((p == 0) | (p == 1)).then(0.0).otherwise(-p * p.log() - q * q.log())
