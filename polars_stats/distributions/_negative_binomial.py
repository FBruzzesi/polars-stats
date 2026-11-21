from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from polars_stats.distributions._base import DiscreteDistribution, coerce_param, scalar_float, scalar_kwargs

if TYPE_CHECKING:
    import polars as pl

    from polars_stats._typing import DistributionName, IntoExprColumn


class NegativeBinomial(DiscreteDistribution):
    """Negative binomial distribution: the number of failures before the ``r``-th success.

    Each trial succeeds with probability ``p``. Equivalent to ``scipy.stats.nbinom(n=r, p=p)``; ``statrs`` also names
    the first parameter ``r``. The support is the non-negative integers, counting failures from ``0``, unlike
    ``Geometric``, which counts trials from ``1``: ``NegativeBinomial(1, p)`` at ``k`` is ``Geometric(p)`` at
    ``k + 1``. ``r`` need not be an integer, which is the Polya extension used for overdispersed counts.

    Arguments:
        r: Number of successes, with ``r > 0`` and not necessarily an integer. Either a Python ``float`` or an
            ``IntoExprColumn`` (``pl.Expr``, ``pl.Series`` or column name ``str``) carrying one value per row.
        p: Success probability of each trial, with ``0 < p <= 1``. Same accepted types as ``r``.

    An invalid parameterisation (``r <= 0``, a ``p`` outside ``(0, 1]`` or a non-finite parameter) is not checked at
    construction; it raises ``InvalidOperation`` (a ``ComputeError``) when any method is evaluated. Below the support
    ``pmf`` and ``cdf`` are ``0`` and ``sf`` is ``1``; ``pmf`` is ``0`` at a non-integral point. A null parameter nulls
    every method, on the support and off it. Samples are ``UInt64`` counts, so ``sample`` and ``samples`` refuse a
    mean ``r (1 - p) / p`` above ``1.844e19``, and raise if a draw's Gamma rate passes it. ``entropy`` has no closed
    form and raises ``NotImplementedError``.
    """

    _r: pl.Expr
    _p: pl.Expr
    _distribution_name: ClassVar[DistributionName] = "negativebinomial"

    def __init__(self, r: float | IntoExprColumn, p: float | IntoExprColumn) -> None:
        self._r = coerce_param(r, name="r")
        self._p = coerce_param(p, name="p")
        self._scalar_kwargs = scalar_kwargs(r=scalar_float(r), p=scalar_float(p))

    @property
    def _param_exprs(self) -> tuple[pl.Expr, ...]:
        return (self._r, self._p)

    @property
    def _validated_params(self) -> pl.Expr:
        return self._validated("p", self._p)

    def mean(self) -> pl.Expr:
        """Expected value, ``r * (1 - p) / p``."""
        return self._moment(self._r * (1 - self._p) / self._p)

    def variance(self) -> pl.Expr:
        """Variance, ``r * (1 - p) / p**2``."""
        return self._moment(self._r * (1 - self._p) / self._p**2)

    def std(self) -> pl.Expr:
        """Standard deviation, ``sqrt(r * (1 - p)) / p``.

        Not ``variance().sqrt()``: squaring ``p`` overflows about 150 decades earlier.
        """
        return self._moment((self._r * (1 - self._p)).sqrt() / self._p)

    def entropy(self) -> pl.Expr:
        """Shannon entropy: not implemented.

        It has no closed form, and the support sum costs ``O(sd)`` terms per row, ``1e6`` at ``p = 1e-6``.

        Raises:
            NotImplementedError: Always.
        """
        msg = "NegativeBinomial.entropy has no closed form and is not implemented"
        raise NotImplementedError(msg)
