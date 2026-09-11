from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import polars as pl

from polars_stats.distributions._base import DiscreteDistribution, coerce_param, scalar_float, scalar_kwargs

if TYPE_CHECKING:
    from polars_stats._typing import DistributionName, IntoExprColumn


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

    An invalid ``p`` (``p <= 0``, ``p > 1`` or ``NaN``) is not checked at construction; it raises
    ``InvalidOperation`` (a ``ComputeError``) when any method is evaluated. A null ``p`` nulls every method,
    on the support and off it. Samples are ``UInt64`` trial counts, so unlike ``Bernoulli`` the degenerate
    ``p = 0`` point mass is not representable.
    """

    _p: pl.Expr
    _distribution_name: ClassVar[DistributionName] = "geometric"

    def __init__(self, p: float | IntoExprColumn) -> None:
        self._p = coerce_param(p, name="p")
        self._scalar_kwargs = scalar_kwargs(p=scalar_float(p))

    @property
    def _param_exprs(self) -> tuple[pl.Expr, ...]:
        return (self._p,)

    @property
    def _validated_params(self) -> pl.Expr:
        return self._validated("p", self._p)

    @property
    def _raw_p(self) -> pl.Expr:
        """``p`` as ``Float64`` without the validating round trip, so a moment can name it repeatedly."""
        return self._p.cast(pl.Float64())

    def mean(self) -> pl.Expr:
        """Expected value, ``1 / p``."""
        return 1 / self._validated_params

    def variance(self) -> pl.Expr:
        """Variance, ``(1 - p) / p**2``."""
        p = self._raw_p
        return self._moment((1 - p) / p**2)

    def std(self) -> pl.Expr:
        """Standard deviation, ``sqrt(1 - p) / p``.

        Not ``variance().sqrt()``: squaring and unsquaring ``p`` overflows about 150 decades earlier.
        """
        p = self._raw_p
        return self._moment((1 - p).sqrt() / p)

    def entropy(self) -> pl.Expr:
        """Shannon entropy, ``(-(1 - p) * log1p(-p) - p * log(p)) / p``, with ``0 * log 0 = 0`` at ``p = 1``."""
        p = self._raw_p
        q = 1 - p
        shannon = pl.when(p == 1).then(0.0).otherwise((-q * (-p).log1p() - p * p.log()) / p)
        return self._moment(shannon)
