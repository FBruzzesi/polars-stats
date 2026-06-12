from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import polars as pl

from polars_stats.distributions._base import (
    ROW_INDEX_EXPR,
    DiscreteDistribution,
    coerce_param,
    register_plugin,
    scalar_float,
    scalar_kwargs,
)

if TYPE_CHECKING:
    from polars_stats._typing import IntoExprColumn


class Bernoulli(DiscreteDistribution):
    """Bernoulli distribution with success probability ``p``.

    Arguments:
        p: Success probability of Bernoulli distribution. Either a Python ``float`` or an ``IntoExprColumn``
            (``pl.Expr``, ``pl.Series`` or column name ``str``) carrying one probability per row.
    """

    _p: pl.Expr
    _samples_scalar_plugin: ClassVar[str] = "bernoulli_samples_scalar"

    def __init__(self, p: float | IntoExprColumn) -> None:
        self._p = coerce_param(p, name="p")
        # A constant `p` enables the fast sampler path; `None` falls back to the per-row plugin.
        self._scalar_kwargs = scalar_kwargs(p=scalar_float(p))

    @property
    def _checked_p(self) -> pl.Expr:
        """``p`` validated in Rust to lie in ``[0, 1]`` (raises otherwise).

        Computed in Rust so the closed-form methods report an invalid ``p`` consistently with
        ``sample`` rather than silently computing a negative probability. Null propagates.
        """
        return register_plugin("bernoulli_proba", self._p)

    def sample(self, seed: int | None = None) -> pl.Expr:
        """Draw one Bernoulli sample per row, returning a ``Boolean`` column.

        Output length follows the surrounding context (frame length under ``select`` / ``with_columns``,
        partition length under ``over`` / ``group_by``). Each row's draw is derived from a per-row sub-seed mixed from
        ``seed`` and the row's position, so the result is independent of Polars chunking and thread scheduling.

        Rows with an invalid parameter raise; rows with a null parameter yield null.

        The output column is named ``"sample"`` when ``p`` is a constant; a column-valued ``p``
        keeps polars root-name semantics (the name follows the parameter expression).
        """
        if self._scalar_kwargs is not None:
            return register_plugin(
                "bernoulli_sample_scalar", (ROW_INDEX_EXPR,), kwargs={"seed": seed, **self._scalar_kwargs}
            ).alias("sample")
        return register_plugin("bernoulli_sample", (self._p, ROW_INDEX_EXPR), kwargs={"seed": seed})

    def _samples_columns(self, size: int, seed: int | None) -> pl.Expr:
        return register_plugin("bernoulli_samples", (self._p, ROW_INDEX_EXPR), kwargs={"seed": seed, "size": size})

    def _pmf(self, value: pl.Expr) -> pl.Expr:
        """``1 - p`` at 0, ``p`` at 1, ``0`` elsewhere."""
        p = self._checked_p
        return pl.when(value.eq(0)).then(1 - p).when(value.eq(1)).then(p).otherwise(0.0)

    def _cdf(self, value: pl.Expr) -> pl.Expr:
        """``0`` for ``value < 0``, ``1 - p`` for ``0 <= value < 1``, ``1`` for ``value >= 1``."""
        return pl.when(value.lt(0)).then(0.0).when(value.lt(1)).then(1 - self._checked_p).otherwise(1.0)

    def _ppf(self, quantile: pl.Expr) -> pl.Expr:
        """Smallest ``x`` with ``cdf(x) >= quantile``: ``0`` if ``quantile <= 1 - p`` else ``1`` (``Boolean``)."""
        return quantile > 1 - self._checked_p

    def mean(self) -> pl.Expr:
        """Expected value, ``p``."""
        return self._checked_p

    def variance(self) -> pl.Expr:
        """Variance, ``p * (1 - p)``."""
        p = self._checked_p
        return p * (1 - p)

    def entropy(self) -> pl.Expr:
        """Shannon entropy, ``-p * log(p) - (1 - p) * log(1 - p)``.

        Uses the convention ``0 * log 0 = 0`` so the result is ``0`` at the degenerate endpoints ``p in {0, 1}``.
        """
        p = self._checked_p
        q = 1 - p

        return pl.when((p == 0) | (p == 1)).then(0.0).otherwise(-p * p.log() - q * q.log())
