from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import polars as pl

from polars_stats.distributions._base import (
    ROW_INDEX_EXPR,
    DiscreteDistribution,
    coerce_int,
    register_plugin,
    scalar_int,
    scalar_kwargs,
)

if TYPE_CHECKING:
    from polars_stats._typing import IntoExprColumn

_INV_12 = 1 / 12
"""The correctly rounded ``1 / 12``, multiplied rather than divided against so the constant-parameter
and per-row kernels cannot disagree in the last bit (see `n`)."""


class DiscreteUniform(DiscreteDistribution):
    """Discrete uniform distribution over the integers ``{min, ..., max}``, **both bounds inclusive**.

    Equivalent to ``scipy.stats.randint(low=min, high=max + 1)``. The ``max`` argument is
    **inclusive**, unlike scipy's exclusive ``high``: the support is ``{min, ..., max}`` and
    ``cdf(max) == 1``. This is the single most common trap when porting scipy code.

    Arguments:
        min: Inclusive lower bound, an integer. Either a Python ``int`` or an ``IntoExprColumn``
            (``pl.Expr``, ``pl.Series`` or column name ``str``) carrying one bound per row.
        max: Inclusive upper bound, with ``max >= min`` (``min == max`` is a one-point mass). Same
            accepted types as ``min``.

    An invalid parameterisation (``max < min``, or a width ``max - min + 1`` that overflows
    ``Int64``) is not checked at construction; matching every other distribution, it raises
    ``InvalidOperation`` (a ``ComputeError``) when any method is evaluated. Null bounds propagate
    to null.
    """

    _min: pl.Expr
    _max: pl.Expr
    _plugin_prefix: ClassVar[str] = "discreteuniform"

    def __init__(self, min: int | IntoExprColumn, max: int | IntoExprColumn) -> None:  # noqa: A002
        self._min = coerce_int(min, name="min")
        self._max = coerce_int(max, name="max")
        self._scalar_kwargs = scalar_kwargs(min=scalar_int(min), max=scalar_int(max))

    @property
    def _param_exprs(self) -> tuple[pl.Expr, ...]:
        return (self._min, self._max)

    @property
    def n(self) -> pl.Expr:
        """Support count ``N = max - min + 1``, computed by the Rust range validator as ``Float64``.

        Every closed-form method divides by this, so the validator is what makes them raise on an
        invalid parameterisation (``max < min``, an overflowing width) consistently with the sampler;
        null bounds propagate. Both the constant-parameter and the per-row routing consume the
        validator's own output rather than a Polars-recomputed fallback: with literals, folding
        ``max - min + 1`` into a constant lets the engine turn those divisions into reciprocal
        multiplies, which drifts the scalar path a few ulps from the per-row kernel and breaks
        fast-path bit-equality on exactly the methods this distribution exists to provide.
        """
        if self._scalar_kwargs is None:
            return register_plugin("discreteuniform_range", self._param_exprs)
        return register_plugin("discreteuniform_range", self._scalar_lit_args())

    @property
    def _checked_params(self) -> pl.Expr:
        """The validated support count; see `n`."""
        return self.n

    @property
    def _inv_n(self) -> pl.Expr:
        """``1 / n``, read off the validator once so every consumer multiplies by the same rounded reciprocal.

        Dividing by ``n`` directly is not bit-stable across the two parameter routings: polars
        divides a column by a broadcast scalar through a different code path than column by column,
        and the two disagree in the last bit. Multiplying by a single materialised ``1 / n`` removes
        the fork -- every method below performs the very same multiplication on both paths.
        """
        return 1 / self.n

    def _on_support(self, value: pl.Expr) -> pl.Expr:
        """Whether ``value`` is an integer inside ``[min, max]``, where the mass sits."""
        return (value >= self._min) & (value <= self._max) & (value.floor() == value)

    def _pmf(self, value: pl.Expr) -> pl.Expr:
        """``1 / N`` on the integer support, ``0`` off it, ``null`` for null bounds.

        The second ``when`` arm is load bearing: ``on_support`` reads the bounds, so a null bound
        makes its condition -- not its result -- null, and an unmatched ``when`` arm would silently
        route a null-bounds row into the ``0.0`` off-support answer. Re-checking ``n``'s nullness
        turns those rows back into nulls, on both engines.
        """
        return (
            pl.when(self._on_support(value))
            .then(self._inv_n)
            .when(self.n.is_not_null())
            .then(0.0)
            .otherwise(pl.lit(None, dtype=pl.Float64()))
        )

    def _sf(self, value: pl.Expr) -> pl.Expr:
        """``(max - floor(value)) * (1 / N)`` inside the support, ``1`` below ``min``, ``0`` from ``max`` up.

        A direct count of the support points above ``value``, mirroring `_cdf`. Overrides the base
        ``1 - cdf``, whose subtraction absorbs the whole tail once the cdf rounds towards ``1`` --
        a few parts in ``1e11`` of relative error at ``N ~ 1e6``, where the true sf is ``1 / N``.

        The whole chain sits behind a ``n.is_not_null()`` gate: the ``1.0`` below-support branch reads
        only ``min``, so without the gate a null ``max`` would answer ``1.0`` instead of propagating.
        """
        return (
            pl.when(self.n.is_not_null())
            .then(
                pl.when(value < self._min)
                .then(1.0)
                .when(value >= self._max)
                .then(0.0)
                .otherwise((self._max.cast(pl.Float64()) - value.floor()) * self._inv_n)
            )
            .otherwise(pl.lit(None, dtype=pl.Float64()))
        )

    def _log_pmf(self, value: pl.Expr) -> pl.Expr:
        """``-log(N)`` on the integer support, ``-inf`` off it, ``null`` for null bounds.

        Overrides the base ``pmf().log()`` only to skip the pointless division: the mass is exactly
        ``1 / N``, so its log reads straight off the count. The extra ``when`` arm preserves null
        bounds for the reason in `_pmf`.
        """
        return (
            pl.when(self._on_support(value))
            .then(-self.n.log())
            .when(self.n.is_not_null())
            .then(float("-inf"))
            .otherwise(pl.lit(None, dtype=pl.Float64()))
        )

    def _cdf(self, value: pl.Expr) -> pl.Expr:
        """``(floor(value) - min + 1) * (1 / N)`` clamped to ``[0, 1]``.

        The floor counts the support points at or below ``value``; clamping covers everything below
        ``min`` (a non-positive numerator) and above ``max`` (a numerator past ``N``), so
        ``cdf(max) == 1`` exactly -- the inclusive upper bound is the visible difference from scipy's
        exclusive one. The count enters through `_inv_n`; see that property for why this is a
        multiplication rather than a division.
        """
        return ((value.floor() - self._min + 1) * self._inv_n).clip(0.0, 1.0)

    def _ppf(self, quantile: pl.Expr) -> pl.Expr:
        """``min + ceil(quantile * N) - 1``, corrected and clamped; null outside ``[0, 1]``.

        The closed form matches scipy's ``randint.ppf`` -- not just the ceiling formula but its
        correction step, which probes ``k - 1`` (clamped into the support) and keeps it whenever its
        cdf already reaches ``quantile``. That probe is what settles the step-boundary cases: an
        ulp of noise in ``quantile * N`` around an exact integer would otherwise skip a support
        point, and matching scipy means resolving them the same way rather than hoping the grids
        miss them.

        The count enters through `n_rowwise` -- validated at the frame's full height, not broadcast
        from a length-1 value. The correction's ``>=`` has no slack, so it needs the honestly
        rounded quotient of a column-by-column division: polars rewrites a division by a broadcast
        scalar as a reciprocal multiply, whose last bit lands on either side of the step depending
        on the engine.
        """
        if self._scalar_kwargs is None:
            n_rowwise = register_plugin("discreteuniform_range_rowwise", [*self._param_exprs, ROW_INDEX_EXPR])
        else:
            n_rowwise = register_plugin("discreteuniform_range_rowwise", [*self._scalar_lit_args(), ROW_INDEX_EXPR])
        lo = self._min.cast(pl.Float64())
        hi = self._max.cast(pl.Float64())
        candidate = lo + (quantile * n_rowwise).ceil() - 1.0
        stepped = (candidate - 1.0).clip(lo, hi)
        return (
            pl.when(quantile.is_between(0, 1))
            .then(
                pl.when((stepped.floor() - self._min + 1) / n_rowwise >= quantile)
                .then(stepped)
                .otherwise(candidate)
                .clip(lo, hi)
            )
            .otherwise(pl.lit(None, dtype=pl.Float64()))
        )

    def _isf(self, quantile: pl.Expr) -> pl.Expr:
        """``max - floor(quantile * N)``, clamped to the support; null outside ``[0, 1]``.

        Entered against ``quantile`` itself, not the base ``ppf(1 - quantile)``: the complement
        rounds before the inverse runs, and on this distribution that flips whole support points --
        the survival steps sit at multiples of ``1 / N``, exactly where ``1 - q`` has spent its
        precision. In real arithmetic this is the same function (smallest ``k`` with
        ``(max - k) / N <= q``), and at ``q = 0`` it reads ``max`` directly, where ``sf(max) = 0``
        already satisfies the inequality.
        """
        lo = self._min.cast(pl.Float64())
        hi = self._max.cast(pl.Float64())
        return (
            pl.when(quantile.is_between(0, 1))
            .then((hi - (quantile * self.n).floor()).clip(lo, hi))
            .otherwise(pl.lit(None, dtype=pl.Float64()))
        )

    def mean(self) -> pl.Expr:
        """Expected value, ``(min + max) / 2``."""
        return self._moment((self._min + self._max) / 2)

    def variance(self) -> pl.Expr:
        """Variance, ``(N**2 - 1) / 12``."""
        return self._moment((self.n**2 - 1) * _INV_12)

    def median(self) -> pl.Expr:
        """Median, ``(min + max) / 2``.

        Overrides the base ``ppf(0.5)``, which would answer a support point: for even ``N`` the
        midpoint falls between two of them, and it is the midpoint scipy reports too.
        """
        return self._moment((self._min + self._max) / 2)

    def entropy(self) -> pl.Expr:
        """Shannon entropy in nats, ``log(N)``.

        statrs implements no entropy for this distribution, but every outcome carries mass ``1 / N``,
        so the support sum collapses to the elementary form and lives here rather than in Rust.
        """
        return self._moment(self.n.log())
