from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from polars_stats.distributions._base import (
    DiscreteDistribution,
    coerce_n,
    coerce_param,
    scalar_float,
    scalar_int,
    scalar_kwargs,
)

if TYPE_CHECKING:
    import polars as pl

    from polars_stats._typing import DistributionName, IntoExprColumn


class Binomial(DiscreteDistribution):
    """Binomial distribution: number of successes in ``n`` trials, each with success probability ``p``.

    Equivalent to ``scipy.stats.binom(n, p)``. The argument order differs from ``statrs``
    (``Binomial(p, n)``); this class follows scipy's ``(n, p)``.

    Arguments:
        n: Number of trials, an integer ``>= 0``. Either a Python ``int`` (at most ``2**63 - 1``) or an
            ``IntoExprColumn`` (``pl.Expr``, ``pl.Series`` or column name ``str``) carrying one count per
            row. A column must have an integer dtype, of any width up to ``UInt64``, and may hold any count
            the dtype can; cast a float one yourself (``pl.col("n").cast(pl.Int64)``).
        p: Success probability in ``[0, 1]``. Either a Python ``float`` or an ``IntoExprColumn``.

    Parameters are validated at evaluation, where a negative ``n`` column, a non-integer ``n`` dtype or a
    ``p`` outside ``[0, 1]`` raises ``InvalidOperation`` (a ``ComputeError``). A scalar ``n`` is the one
    exception: it is coerced to a ``UInt64`` literal and passed to the fast paths as a kwarg, neither of
    which can carry an out-of-range count, so it raises ``ValueError`` at construction. Construction
    otherwise rejects only wrong *types* (``TypeError``). Null parameters propagate to null, a
    ``Null``-dtype ``n`` column included; the dtype rule is judged first, so a *float* ``n`` column
    raises even when every value in it is null.
    """

    _n: pl.Expr
    _p: pl.Expr
    _distribution_name: ClassVar[DistributionName] = "binomial"

    def __init__(self, n: int | IntoExprColumn, p: float | IntoExprColumn) -> None:
        self._n = coerce_n(n, name="n")
        self._p = coerce_param(p, name="p")
        self._scalar_kwargs = scalar_kwargs(n=scalar_int(n), p=scalar_float(p))

    @property
    def _param_exprs(self) -> tuple[pl.Expr, ...]:
        return (self._n, self._p)

    @property
    def _validated_params(self) -> pl.Expr:
        return self._validated("params", self._p)

    def _log_cdf(self, value: pl.Expr) -> pl.Expr:
        """No Rust ``ln_cdf`` body yet, so this underflows to ``-inf`` where ``cdf`` rounds to ``0``."""
        return self._cdf(value).log()

    def _log_sf(self, value: pl.Expr) -> pl.Expr:
        """No Rust ``ln_sf`` body yet, so this underflows to ``-inf`` where ``sf`` rounds to ``0``."""
        return self._sf(value).log()

    def mean(self) -> pl.Expr:
        """Expected value, ``n * p``."""
        return self._moment(self._n * self._p)

    def variance(self) -> pl.Expr:
        """Variance, ``n * p * (1 - p)``."""
        return self._moment(self._n * self._p * (1 - self._p))

    def entropy(self) -> pl.Expr:
        """Shannon entropy in nats, the exact support sum ``-sum_k pmf(k) log pmf(k)``.

        ``0`` at the degenerate endpoints ``p in {0, 1}``. There is no closed form, so ``binomial_entropy``
        evaluates the sum in Rust.
        """
        return self._param_plugin("entropy")
