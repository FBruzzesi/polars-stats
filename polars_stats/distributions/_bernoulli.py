from __future__ import annotations

import polars as pl
from polars.plugins import register_plugin_function

from polars_stats._lib import LIB


class Bernoulli:
    """Bernoulli distribution with success probability ``p``.

    Arguments:
        p: Success probability of Bernoulli distribution.
    """

    _p: pl.Expr

    def __init__(self, p: float | pl.Expr) -> None:
        if isinstance(p, pl.Expr):
            self._p: pl.Expr = p
        elif isinstance(p, float):
            if not 0.0 <= p <= 1.0:
                msg = f"p must be in the [0, 1] range, found {p}"
                raise ValueError(msg)
            # Materialise scalar p to one row per output row.
            # `is_elementwise=True` does NOT broadcast inputs before the plugin runs
            # (it only broadcasts the *result*), so a length-1 `pl.lit(p)` would make
            # the plugin draw a single sample which polars then repeats across the whole frame.
            # `pl.repeat` produces a length-`pl.len()` series so each row gets its own draw.
            self._p = pl.repeat(p, n=pl.len())
        else:
            msg = f"p should be either a pl.Expr or float, found {type(p)}"
            raise TypeError(msg)

    def sample(self, seed: int | None = None) -> pl.Expr:
        """Draw one Bernoulli sample per row, returning a ``UInt8`` 0/1 column.

        Output length follows the surrounding context (frame length under
        ``with_columns`` / ``select``). Multi-output column expressions
        (e.g. ``pl.col("p1", "p2")``) are expanded by polars into one
        plugin call per column. Reproducibility under ``seed`` assumes a
        single-chunk input series; chunked / streaming inputs are not
        supported.
        """
        return register_plugin_function(
            args=[self._p],
            plugin_path=LIB,
            function_name="bernoulli_sample",
            kwargs={"seed": seed},
            is_elementwise=True,
        )
