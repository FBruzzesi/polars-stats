from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from polars_stats.distributions._base import DiscreteDistribution, coerce_param, scalar_float, scalar_kwargs

if TYPE_CHECKING:
    import polars as pl

    from polars_stats._typing import DistributionName, IntoExprColumn


class Poisson(DiscreteDistribution):
    """Poisson distribution: the number of events in a unit interval of a process with rate ``rate``.

    Equivalent to ``scipy.stats.poisson(mu=rate)``. The rate λ is ``rate`` here, as in ``Exponential`` and ``Gamma``,
    so ``Poisson(rate=r)`` counts what ``Exponential(rate=r)`` spaces out; scipy calls it ``mu`` and ``statrs``
    ``lambda``. The support is the non-negative integers.

    Arguments:
        rate: Rate parameter λ, the mean number of events, with ``rate > 0``. Either a Python ``float`` or an
            ``IntoExprColumn`` (``pl.Expr``, ``pl.Series`` or column name ``str``) carrying one rate per row.

    An invalid ``rate`` (``rate <= 0`` or a non-finite value) is not checked at construction; it raises
    ``InvalidOperation`` (a ``ComputeError``) when any method is evaluated. Unlike scipy, ``rate = 0`` is invalid.
    Below the support ``pmf`` and ``cdf`` are ``0`` and ``sf`` is ``1``; ``pmf`` is ``0`` at a non-integral point.
    A null ``rate`` nulls every method, on the support and off it. Samples are ``UInt64`` counts, so ``sample`` and
    ``samples`` also refuse a ``rate`` above ``1.844e19``, where a draw no longer fits one.
    """

    _rate: pl.Expr
    _distribution_name: ClassVar[DistributionName] = "poisson"

    def __init__(self, rate: float | IntoExprColumn) -> None:
        self._rate = coerce_param(rate, name="rate")
        self._scalar_kwargs = scalar_kwargs(rate=scalar_float(rate))

    @property
    def _param_exprs(self) -> tuple[pl.Expr, ...]:
        return (self._rate,)

    @property
    def _validated_params(self) -> pl.Expr:
        return self._validated("rate", self._rate)

    def mean(self) -> pl.Expr:
        """Expected value, ``rate``."""
        return self._validated_params

    def variance(self) -> pl.Expr:
        """Variance, ``rate``."""
        return self._validated_params

    def entropy(self) -> pl.Expr:
        """Shannon entropy in nats, the support sum ``-sum_k pmf(k) log pmf(k)``.

        It has no closed form, so ``poisson_entropy`` sums it in Rust below ``rate = 1000`` and takes its asymptotic
        series in ``1 / rate`` above.
        """
        return self._param_plugin("entropy")
