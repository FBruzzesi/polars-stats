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

    An invalid ``p`` (``p <= 0``, ``p > 1`` or ``NaN``) is not checked at construction; matching every
    other distribution, it raises ``InvalidOperation`` (a ``ComputeError``) when any method is
    evaluated. A null ``p`` propagates to null wherever the result depends on ``p``; the off-support
    constants (``pmf(0) = 0``, ``cdf(0) = 0``, ``sf(0) = 1``, and ``pmf`` at any non-integral point)
    do not, as in ``Bernoulli``. Samples are ``UInt64`` trial counts, so unlike ``Bernoulli`` the
    degenerate ``p = 0`` point mass is not representable.

    The value-keyed methods compute in Rust, so an invalid ``p`` is reported whichever branch the
    value selects. The moments stay in Polars, reading ``p`` through the same Rust validator.
    """

    _p: pl.Expr
    _plugin_prefix: ClassVar[str] = "geometric"

    def __init__(self, p: float | IntoExprColumn) -> None:
        self._p = coerce_param(p, name="p")
        self._scalar_kwargs = scalar_kwargs(p=scalar_float(p))

    @property
    def _param_exprs(self) -> tuple[pl.Expr, ...]:
        return (self._p,)

    @property
    def _checked_p(self) -> pl.Expr:
        """``p`` validated in Rust to lie in ``(0, 1]`` (raises otherwise), null when ``p`` is null.

        Read only by the moments; the value-keyed methods validate inside their own plugin.
        """
        return self._checked("geometric_p", self._p)

    @property
    def _raw_p(self) -> pl.Expr:
        """``p`` as ``Float64`` without the validating round trip, so a moment can name it repeatedly.

        The cast is the one `geometric_p` performs on the way in, so a ``Float32`` parameter column
        still yields ``Float64`` results.
        """
        return self._p.cast(pl.Float64())

    def _when_p_valid(self, formula: pl.Expr) -> pl.Expr:
        """``formula`` behind a single validation of ``p``: raises on an invalid ``p``, null on a null one.

        Polars folds neither a repeated subexpression nor a plugin call, so a moment naming
        `_checked_p` inline would cross FFI once per mention (6x on ``entropy`` over a column ``p``).
        """
        return pl.when(self._checked_p.is_not_null()).then(formula)

    def _pmf(self, value: pl.Expr) -> pl.Expr:
        """``(1 - p)**(k - 1) * p`` on the positive integers, ``0`` elsewhere."""
        return self._value_plugin("geometric_pmf", value)

    def _log_pmf(self, value: pl.Expr) -> pl.Expr:
        """``(k - 1) * log1p(-p) + log(p)`` on the positive integers, ``-inf`` elsewhere."""
        return self._value_plugin("geometric_ln_pmf", value)

    def _cdf(self, value: pl.Expr) -> pl.Expr:
        """``1 - (1 - p)**floor(k)`` from ``1`` up, ``0`` below."""
        return self._value_plugin("geometric_cdf", value)

    def _log_cdf(self, value: pl.Expr) -> pl.Expr:
        """``log(1 - (1 - p)**floor(k))`` from ``1`` up, ``-inf`` below."""
        return self._value_plugin("geometric_ln_cdf", value)

    def _sf(self, value: pl.Expr) -> pl.Expr:
        """``(1 - p)**floor(k)`` from ``1`` up, ``1`` below."""
        return self._value_plugin("geometric_sf", value)

    def _log_sf(self, value: pl.Expr) -> pl.Expr:
        """``floor(k) * log1p(-p)`` from ``1`` up, ``0`` below."""
        return self._value_plugin("geometric_ln_sf", value)

    def _ppf(self, quantile: pl.Expr) -> pl.Expr:
        """Smallest ``k`` with ``cdf(k) >= quantile``; null for ``quantile`` outside ``[0, 1]``."""
        return self._value_plugin("geometric_ppf", quantile)

    def _isf(self, quantile: pl.Expr) -> pl.Expr:
        """Smallest ``k`` with ``sf(k) <= quantile``; null for ``quantile`` outside ``[0, 1]``."""
        return self._value_plugin("geometric_isf", quantile)

    def mean(self) -> pl.Expr:
        """Expected value, ``1 / p``. The one formula naming ``p`` once, so it reads `_checked_p` directly."""
        return 1 / self._checked_p

    def variance(self) -> pl.Expr:
        """Variance, ``(1 - p) / p**2``."""
        p = self._raw_p
        return self._when_p_valid((1 - p) / p**2)

    def std(self) -> pl.Expr:
        """Standard deviation, ``sqrt(1 - p) / p``.

        Overrides the base-class ``variance().sqrt()``, which squares ``p`` and then unsquares it:
        the round trip overflows about 150 decades before ``sqrt(1 - p) / p`` does.
        """
        p = self._raw_p
        return self._when_p_valid((1 - p).sqrt() / p)

    def entropy(self) -> pl.Expr:
        """Shannon entropy, ``(-(1 - p) * log1p(-p) - p * log(p)) / p``, with the ``0`` limit at ``p = 1``.

        Uses the convention ``0 * log 0 = 0`` so the degenerate ``p = 1`` point mass has entropy ``0``.
        """
        p = self._raw_p
        q = 1 - p
        shannon = pl.when(p == 1).then(0.0).otherwise((-q * (-p).log1p() - p * p.log()) / p)
        return self._when_p_valid(shannon)
