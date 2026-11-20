from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from polars_stats.distributions._base import ContinuousDistribution, coerce_param, scalar_float, scalar_kwargs

if TYPE_CHECKING:
    import polars as pl

    from polars_stats._typing import DistributionName, IntoExprColumn


class ChiSquared(ContinuousDistribution):
    """Chi-squared distribution with ``dof`` degrees of freedom, on the support ``x >= 0``.

    Equivalent to ``scipy.stats.chi2(df=dof)``; ``statrs`` names the parameter ``freedom``. It is
    ``Gamma(shape=dof / 2, rate=1 / 2)``, and every value-keyed method, ``std``, ``entropy`` and seeded samples agree
    with that ``Gamma`` to the last bit. ``variance`` is the exact ``2 * dof``.

    Arguments:
        dof: Degrees of freedom, with ``dof > 0`` and not necessarily an integer. Either a Python ``float`` or an
            ``IntoExprColumn`` (``pl.Expr``, ``pl.Series`` or column name ``str``) carrying one ``dof`` per row.

    An invalid ``dof`` (``dof <= 0`` or a non-finite value) is not checked at construction; it raises
    ``InvalidOperation`` (a ``ComputeError``) when any method is evaluated. Below the support ``pdf`` and ``cdf``
    are ``0`` and ``sf`` is ``1``. A null ``dof`` nulls every method, on the support and below it.
    """

    _dof: pl.Expr
    _distribution_name: ClassVar[DistributionName] = "chisquared"

    def __init__(self, dof: float | IntoExprColumn) -> None:
        self._dof = coerce_param(dof, name="dof")
        self._scalar_kwargs = scalar_kwargs(dof=scalar_float(dof))

    @property
    def _param_exprs(self) -> tuple[pl.Expr, ...]:
        return (self._dof,)

    @property
    def _validated_params(self) -> pl.Expr:
        return self._validated("dof", self._dof)

    def mean(self) -> pl.Expr:
        """Expected value, ``dof``."""
        return self._validated_params

    def variance(self) -> pl.Expr:
        """Variance, ``2 * dof``."""
        return self._validated_params * 2.0

    def std(self) -> pl.Expr:
        """Standard deviation, ``sqrt(2 * dof)``.

        Computed as ``2 * sqrt(dof / 2)``: only the root rounds, and it stays finite where ``2 * dof`` overflows.
        """
        return (self._validated_params * 0.5).sqrt() * 2

    def entropy(self) -> pl.Expr:
        """Differential entropy, ``dof / 2 + log 2 + log Gamma(dof / 2) + (1 - dof / 2) * digamma(dof / 2)``.

        Digamma has no Polars expression, so ``chisquared_entropy`` evaluates it in Rust.
        """
        return self._param_plugin("entropy")
