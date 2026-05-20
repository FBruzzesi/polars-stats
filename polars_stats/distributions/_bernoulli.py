from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from polars_stats._lib import LIB
from polars_stats.distributions._base import DiscreteDistribution

if TYPE_CHECKING:
    from polars_stats._typing import IntoExprColumn


class Bernoulli(DiscreteDistribution):
    """Bernoulli distribution with success probability ``p``.

    Arguments:
        p: Success probability of Bernoulli distribution. Either a Python ``float`` or an ``IntoExprColumn``
            (``pl.Expr``, ``pl.Series`` or column name ``str``) carrying one probability per row.
    """

    _p: pl.Expr

    def __init__(self, p: float | IntoExprColumn) -> None:
        if isinstance(p, float):
            # Expand the scalar to a length-N expression so the plugin always receives a row-aligned input.
            # This lets the call stay `is_elementwise=True`, which is what makes `over` / `group_by`
            # invoke the function once per partition rather than treating it as an aggregation.
            self._p = pl.repeat(p, n=pl.len(), dtype=pl.Float64())
        elif isinstance(p, pl.Expr):
            self._p = p
        elif isinstance(p, pl.Series):
            self._p = pl.lit(p)
        elif isinstance(p, str):
            self._p = pl.col(p)
        else:
            msg = f"p should be a float or IntoExprColumn (pl.Expr, str, pl.Series), found {type(p)}"
            raise TypeError(msg)

    def sample(self, seed: int | None = None) -> pl.Expr:
        """Draw one Bernoulli sample per row, returning a ``Boolean`` column.

        Output length follows the surrounding context:

        * frame length under ``with_columns`` / ``select``
        * partition length under ``over`` / ``group_by``
        """
        return register_plugin_function(
            args=[self._p],
            plugin_path=LIB,
            function_name="bernoulli_sample",
            kwargs={"seed": seed},
            is_elementwise=True,
        )

    def samples(self, size: int, seed: int | None = None) -> pl.Expr:
        """Draw `size` Bernoulli samples per row, returning an expression of type ``Array(Boolean, size)``.

        When ``seed`` is set, distinct sub-seeds are derived from it so the `size` underlying ``sample`` calls
        produce independent streams. Without this, every plugin call would re-seed the same RNG and yield
        ``size`` identical columns.
        """
        return (
            pl.when(self._p.is_not_null())
            .then(self._samples(size=size, seed=seed))
            .otherwise(pl.lit(None, dtype=pl.Array(pl.Boolean(), shape=size)))
        )

    def pmf(self, value: float | pl.Expr) -> pl.Expr:
        """Probability mass function. Returns ``1 - p`` at 0, ``p`` at 1, and ``0`` elsewhere."""
        v = value if isinstance(value, pl.Expr) else pl.lit(value)
        return pl.when(v.eq(0)).then(1 - self._p).when(v.eq(1)).then(self._p).otherwise(0.0)

    def cdf(self, value: float | pl.Expr) -> pl.Expr:
        """Cumulative distribution function.

        Piecewise constant: ``0`` for ``value < 0``, ``1 - p`` for ``0 <= value < 1``, ``1`` for ``value >= 1``.
        """
        v = value if isinstance(value, pl.Expr) else pl.lit(value)
        return pl.when(v.lt(0)).then(0.0).when(v.lt(1)).then(1 - self._p).otherwise(1.0)

    def ppf(self, quantile: float | pl.Expr) -> pl.Expr:
        """Percent point function (inverse cdf).

        Returns the smallest ``x`` such that ``cdf(x) >= quantile``: ``0`` if ``quantile <= 1 - p`` else ``1``,
        as a ``Boolean`` column. Nulls in ``quantile`` are propagated.
        """
        q = quantile if isinstance(quantile, pl.Expr) else pl.lit(quantile)
        return q > 1 - self._p

    def mean(self) -> pl.Expr:
        """Expected value, ``p``."""
        return self._p

    def variance(self) -> pl.Expr:
        """Variance, ``p * (1 - p)``."""
        return self._p * (1 - self._p)

    def entropy(self) -> pl.Expr:
        """Shannon entropy, ``-p * log(p) - (1 - p) * log(1 - p)``.

        Uses the convention ``0 * log 0 = 0`` so the result is ``0`` at the degenerate endpoints ``p in {0, 1}``.
        """
        p = self._p
        q = 1 - p

        return pl.when((p == 0) | (p == 1)).then(0.0).otherwise(-p * p.log() - q * q.log())
