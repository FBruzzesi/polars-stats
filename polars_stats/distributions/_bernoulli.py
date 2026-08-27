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

    def _log_pmf(self, value: pl.Expr) -> pl.Expr:
        """``log1p(-p)`` at 0, ``log(p)`` at 1, ``-inf`` elsewhere.

        Overrides the base ``pmf().log()``, whose ``log(1 - p)`` collapses to ``0.0`` below
        ``p ~ 1.1e-16``. See docs/contributing.md, "Numerical stability".
        """
        p = self._checked_p
        return pl.when(value.eq(0)).then((-p).log1p()).when(value.eq(1)).then(p.log()).otherwise(float("-inf"))

    def _cdf(self, value: pl.Expr) -> pl.Expr:
        """``0`` for ``value < 0``, ``1 - p`` for ``0 <= value < 1``, ``1`` for ``value >= 1``."""
        return pl.when(value.lt(0)).then(0.0).when(value.lt(1)).then(1 - self._checked_p).otherwise(1.0)

    def _log_cdf(self, value: pl.Expr) -> pl.Expr:
        """``-inf`` for ``value < 0``, ``log1p(-p)`` for ``0 <= value < 1``, ``0`` for ``value >= 1``.

        Same rounding as ``_log_pmf``: the base ``cdf().log()`` loses the whole small-``p`` regime.
        """
        return (
            pl.when(value.lt(0)).then(float("-inf")).when(value.lt(1)).then((-self._checked_p).log1p()).otherwise(0.0)
        )

    def _sf(self, value: pl.Expr) -> pl.Expr:
        """``1`` for ``value < 0``, ``p`` for ``0 <= value < 1``, ``0`` for ``value >= 1``.

        Overrides the base ``1 - cdf``, which recomputes ``p`` as ``1 - (1 - p)`` and so quantises it
        to the ``1.1e-16`` spacing of ``1.0``, reaching ``0.0`` below that. ``log_sf`` inherits the
        fix through the base ``sf().log()``, which is exact once ``sf`` is.
        """
        return pl.when(value.lt(0)).then(1.0).when(value.lt(1)).then(self._checked_p).otherwise(0.0)

    def _ppf(self, quantile: pl.Expr) -> pl.Expr:
        """Smallest ``x`` with ``cdf(x) >= quantile``: ``0.0`` if ``quantile <= 1 - p`` else ``1.0``.

        A quantile outside ``[0, 1]`` yields null, the contract ``ppf`` guarantees; the bare
        inequality alone has no notion of being out of range and answered ``0.0`` / ``1.0`` there.
        The comparison's ``Boolean`` is cast to ``Float64`` so ``ppf`` / ``isf`` / ``median`` return the
        support point numerically, matching scipy and leaving the wrapper's ``NaN -> NaN`` contract
        representable (``Boolean`` has no ``NaN``).

        ``quantile == 1`` is mapped separately because ``1 - p`` rounds to exactly ``1.0`` below
        ``p ~ 1.1e-16``, which made the comparison false and answered ``0.0`` where ``1.0`` is the
        only correct answer. Every representable ``quantile < 1`` is safely below the true ``1 - p``
        for such a ``p``, so the branch is needed only at the endpoint; ``p == 0`` keeps ``0.0``.
        """
        p = self._checked_p
        return (
            pl.when(quantile.is_between(0, 1))
            .then(
                pl.when(quantile >= 1.0)
                .then((p > 0.0).cast(pl.Float64()))
                .otherwise((quantile > 1 - p).cast(pl.Float64()))
            )
            .otherwise(pl.lit(None, dtype=pl.Float64()))
        )

    def _isf(self, quantile: pl.Expr) -> pl.Expr:
        """Smallest ``x`` with ``sf(x) <= quantile``: ``1.0`` if ``p > quantile`` else ``0.0``.

        Exact for every ``(p, quantile)``: ``sf(0)`` *is* ``p``, so the test forms no complement.
        The base-class ``ppf(1 - quantile)`` formed two and lost the answer whenever either
        saturated (``Bernoulli(1e-17).isf(1e-20)`` returned ``0.0``, not ``1.0``). Endpoints follow
        from the same comparison (``isf(1) = 0`` always, ``isf(0) = 1`` unless ``p`` is ``0``).
        """
        return (
            pl.when(quantile.is_between(0, 1))
            .then((self._checked_p > quantile).cast(pl.Float64()))
            .otherwise(pl.lit(None, dtype=pl.Float64()))
        )

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
        The second term goes through ``log1p`` for the reason in ``_log_pmf``.
        """
        p = self._checked_p
        q = 1 - p

        return pl.when((p == 0) | (p == 1)).then(0.0).otherwise(-p * p.log() - q * (-p).log1p())
