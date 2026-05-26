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

    _p: float | IntoExprColumn

    def __init__(self, p: float | IntoExprColumn) -> None:
        if isinstance(p, float):
            # Expand the scalar to a length-N expression so the plugin always receives a row-aligned input.
            # This lets the call stay `is_elementwise=True`, which is what makes `over` / `group_by`
            # invoke the function once per partition rather than treating it as an aggregation.
            self._p = pl.repeat(p, n=pl.len(), dtype=pl.Float64())
        elif isinstance(p, (pl.Expr, pl.Series, str)):
            self._p = p
        else:
            msg = f"p should be a float or IntoExprColumn (pl.Expr, str, pl.Series), found {type(p)}"
            raise TypeError(msg)

    def sample(self, seed: int | None = None) -> pl.Expr:
        """Draw one Bernoulli sample per row, returning a ``Boolean`` column.

        Output length follows the surrounding context:

        * frame length under ``with_columns`` / ``select``
        * partition length under ``over`` / ``group_by``

        The plugin is genuinely elementwise: each row's draw is derived from a per-row
        sub-seed mixed from ``seed`` and the row's position in the surrounding context,
        so the result is independent of Polars chunking and thread scheduling.
        """
        row_index = pl.int_range(0, pl.len(), dtype=pl.UInt64).alias("__polars_stats_row_index__")
        return register_plugin_function(
            args=[self._p, row_index],
            plugin_path=LIB,
            function_name="bernoulli_sample",
            kwargs={"seed": seed},
            is_elementwise=True,
        )

    def samples(self, size: int, seed: int | None = None) -> pl.Expr:
        msg = "samples is not yet implemented for Bernoulli"
        raise NotImplementedError(msg)

    def cdf(self, value: float | pl.Expr) -> pl.Expr:
        msg = "cdf is not yet implemented for Bernoulli"
        raise NotImplementedError(msg)

    def ppf(self, quantile: float | pl.Expr) -> pl.Expr:
        msg = "ppf is not yet implemented for Bernoulli"
        raise NotImplementedError(msg)

    def mean(self) -> pl.Expr:
        msg = "mean is not yet implemented for Bernoulli"
        raise NotImplementedError(msg)

    def variance(self) -> pl.Expr:
        msg = "variance is not yet implemented for Bernoulli"
        raise NotImplementedError(msg)

    def entropy(self) -> pl.Expr:
        msg = "entropy is not yet implemented for Bernoulli"
        raise NotImplementedError(msg)

    def pmf(self, value: float | pl.Expr) -> pl.Expr:
        msg = "pmf is not yet implemented for Bernoulli"
        raise NotImplementedError(msg)
