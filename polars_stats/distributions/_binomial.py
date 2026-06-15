from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from polars_stats.distributions._base import (
    DiscreteDistribution,
    coerce_n,
    coerce_param,
    register_plugin,
    scalar_float,
    scalar_int,
    scalar_kwargs,
)

if TYPE_CHECKING:
    import polars as pl

    from polars_stats._typing import IntoExprColumn


class Binomial(DiscreteDistribution):
    """Binomial distribution: number of successes in ``n`` trials, each with success probability ``p``.

    Equivalent to ``scipy.stats.binom(n, p)``. The argument order differs from ``statrs``
    (``Binomial(p, n)``); this class follows scipy's ``(n, p)``.

    Arguments:
        n: Number of trials, an integer ``>= 0``. Either a Python ``int`` or an ``IntoExprColumn``
            (``pl.Expr``, ``pl.Series`` or column name ``str``) carrying one count per row.
        p: Success probability in ``[0, 1]``. Either a Python ``float`` or an ``IntoExprColumn``.

    Neither parameter is validated at construction: a negative ``n`` or a ``p`` outside ``[0, 1]``
    raises ``InvalidOperation`` (a ``ComputeError``) when a method is evaluated, identically to an
    invalid column row. Construction rejects only wrong *types* (``TypeError``). Null parameters
    propagate to null.
    """

    _n: pl.Expr
    _p: pl.Expr
    _plugin_prefix: ClassVar[str] = "binomial"

    def __init__(self, n: int | IntoExprColumn, p: float | IntoExprColumn) -> None:
        self._n = coerce_n(n, name="n")
        self._p = coerce_param(p, name="p")
        # Constant `(n, p)` enables the fast sampler path; `None` falls back to the per-row plugin.
        self._scalar_kwargs = scalar_kwargs(n=scalar_int(n), p=scalar_float(p))

    @property
    def _param_exprs(self) -> tuple[pl.Expr, ...]:
        return (self._n, self._p)

    @property
    def _checked_params(self) -> pl.Expr:
        """``p`` validated in Rust against the full ``(n, p)`` parameterisation (raises otherwise).

        Mirrors ``Normal._checked_params`` / ``Bernoulli._checked_p``: the closed-form moments
        (``mean``, ``variance``) derive from this single FFI round-trip, so they report an invalid
        parameterisation (``n < 0`` or ``p`` outside ``[0, 1]``) as a ``ComputeError`` consistently
        with the value-keyed methods, rather than silently computing a moment from invalid inputs.
        Null in either parameter propagates to null.
        """
        return register_plugin("binomial_params", self._param_exprs)

    def _pmf(self, value: pl.Expr) -> pl.Expr:
        """Mass via native ``Discrete::pmf``; zero off the integer support ``{0, ..., n}``."""
        return self._value_plugin("binomial_pmf", value)

    def _log_pmf(self, value: pl.Expr) -> pl.Expr:
        """Log-mass via native ``Discrete::ln_pmf`` (more accurate than ``pmf().log()``)."""
        return self._value_plugin("binomial_ln_pmf", value)

    def _cdf(self, value: pl.Expr) -> pl.Expr:
        """Cumulative mass ``P(X <= floor(value))`` via native ``DiscreteCDF::cdf``."""
        return self._value_plugin("binomial_cdf", value)

    def _sf(self, value: pl.Expr) -> pl.Expr:
        """Survival ``P(X > floor(value))`` via native ``DiscreteCDF::sf`` (accurate upper tail).

        ``log_sf`` and ``isf`` inherit the base-class defaults, which compose this native ``sf`` and
        ``ppf`` rather than the generic ``1 - cdf`` fallback.
        """
        return self._value_plugin("binomial_sf", value)

    def _ppf(self, quantile: pl.Expr) -> pl.Expr:
        """Inverse cdf via the binary-search ``DiscreteCDF::inverse_cdf``, as an integer-valued ``Float64``.

        A quantile outside ``[0, 1]`` yields null. At the endpoints this returns the support bounds
        (``ppf(0) = 0``, ``ppf(1) = n``); scipy's below-support sentinel ``ppf(0) = -1`` is not
        reproduced. ``median`` is ``ppf(0.5)`` (the base-class default), which matches scipy exactly;
        statrs' native ``floor(n * p)`` median is a different convention and is deliberately not used.
        """
        return self._value_plugin("binomial_ppf", quantile)

    def mean(self) -> pl.Expr:
        """Expected value, ``n * p``."""
        return self._moment(self._n * self._p)

    def variance(self) -> pl.Expr:
        """Variance, ``n * p * (1 - p)``."""
        return self._moment(self._n * self._p * (1 - self._p))

    def entropy(self) -> pl.Expr:
        """Shannon entropy in nats, the exact support sum ``-sum_k pmf(k) log pmf(k)``.

        ``0`` at the degenerate endpoints ``p in {0, 1}``.
        """
        return register_plugin("binomial_entropy", (self._n, self._p))
