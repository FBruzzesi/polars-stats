from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import polars as pl

from polars_stats.distributions._base import DiscreteDistribution, coerce_param, scalar_float, scalar_kwargs

if TYPE_CHECKING:
    from polars_stats._typing import DistributionName, IntoExprColumn


class Bernoulli(DiscreteDistribution):
    """Bernoulli distribution with success probability ``p``.

    Equivalent to ``scipy.stats.bernoulli(p)``.

    Arguments:
        p: Success probability, with ``0 <= p <= 1``. Either a Python ``float`` or an ``IntoExprColumn``
            (``pl.Expr``, ``pl.Series`` or column name ``str``) carrying one probability per row.

    An invalid ``p`` (``p < 0``, ``p > 1`` or ``NaN``) is not checked at construction; it raises
    ``InvalidOperation`` (a ``ComputeError``) when any method is evaluated. A null ``p`` nulls every method,
    on the support and off it.
    """

    _p: pl.Expr
    _distribution_name: ClassVar[DistributionName] = "bernoulli"

    def __init__(self, p: float | IntoExprColumn) -> None:
        self._p = coerce_param(p, name="p")
        self._scalar_kwargs = scalar_kwargs(p=scalar_float(p))

    @property
    def _param_exprs(self) -> tuple[pl.Expr, ...]:
        return (self._p,)

    @property
    def _validated_params(self) -> pl.Expr:
        return self._validated("proba", self._p)

    @property
    def _raw_p(self) -> pl.Expr:
        """``p`` as ``Float64`` without the validating round trip, so a moment can name it repeatedly."""
        return self._p.cast(pl.Float64())

    def mean(self) -> pl.Expr:
        """Expected value, ``p``."""
        return self._validated_params

    def variance(self) -> pl.Expr:
        """Variance, ``p * (1 - p)``."""
        p = self._raw_p
        return self._moment(p * (1 - p))

    def entropy(self) -> pl.Expr:
        """Shannon entropy, ``-p * log(p) - (1 - p) * log1p(-p)``, with ``0 * log 0 = 0`` at ``p in {0, 1}``.

        ``log1p(-p)`` rather than ``log(1 - p)``, which collapses to ``0.0`` below ``p ~ 1.1e-16``.
        """
        p = self._raw_p
        shannon = pl.when((p == 0) | (p == 1)).then(0.0).otherwise(-p * p.log() - (1 - p) * (-p).log1p())
        return self._moment(shannon)
