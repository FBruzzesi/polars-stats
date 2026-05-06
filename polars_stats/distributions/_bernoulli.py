from __future__ import annotations

import polars as pl
from polars.plugins import register_plugin_function

from polars_stats._lib import LIB


class Bernoulli:
    """Bernoulli distribution with success probability ``p``.

    Arguments:
        p: Success probability of Bernoulli distribution. Either a Python
            ``float`` (validated eagerly) or a ``pl.Expr`` carrying one
            probability per row (validated in Rust at sample time).
    """

    _p: pl.Expr

    def __init__(self, p: float | pl.Expr) -> None:
        if isinstance(p, pl.Expr):
            self._p = p
        elif isinstance(p, float):
            if not 0.0 <= p <= 1.0:
                msg = f"p must be in the [0, 1] range, found {p}"
                raise ValueError(msg)
            # Expand the scalar to a length-N expression so the plugin always receives a row-aligned input.
            # This lets the call stay `is_elementwise=True`, which is what makes `over` / `group_by`
            # invoke the function once per partition rather than treating it as an aggregation.
            self._p = pl.repeat(p, n=pl.len(), dtype=pl.Float64())
        else:
            msg = f"p should be either a pl.Expr or float, found {type(p)}"
            raise TypeError(msg)

    def sample(self, seed: int | None = None) -> pl.Expr:
        """Draw one Bernoulli sample per row, returning a ``UInt8`` 0/1 column.

        Output length follows the surrounding context (frame length under
        ``with_columns`` / ``select``, partition length under ``over`` /
        ``group_by``). Reproducibility under ``seed`` assumes a single-chunk
        input series; chunked / streaming inputs are not supported.
        """
        return register_plugin_function(
            args=[self._p],
            plugin_path=LIB,
            function_name="bernoulli_sample",
            kwargs={"seed": seed},
            is_elementwise=True,
        )
