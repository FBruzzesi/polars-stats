from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import polars as pl

from polars_stats.distributions._base import (
    DiscreteDistribution,
    coerce_int,
    scalar_int,
    scalar_kwargs,
)

if TYPE_CHECKING:
    from polars_stats._typing import IntoExprColumn

_ONE_TWELFTH = 1 / 12
"""The correctly rounded ``1 / 12``.

Multiplied by rather than divided against: polars spells ``column / 12`` as an honest division up to
eight rows and as a reciprocal multiply above, so dividing would make ``variance`` row-count
dependent. Costs a third of an ulp against a correctly rounded quotient.
"""


class DiscreteUniform(DiscreteDistribution):
    """Discrete uniform distribution over the integers ``{min, ..., max}``, **both bounds inclusive**.

    Equivalent to ``scipy.stats.randint(low=min, high=max + 1)``. The ``max`` argument is
    **inclusive**, unlike scipy's exclusive ``high``: the support is ``{min, ..., max}`` and
    ``cdf(max) == 1``.

    Two further divergences from scipy: ``median`` is the midpoint ``(min + max) / 2``, not the
    support point scipy's ``ppf(0.5)`` reports, and ``ppf(0)`` / ``isf(1)`` clamp to the support
    rather than answering its below-support sentinel ``min - 1``.

    Arguments:
        min: Inclusive lower bound, an integer. Either a Python ``int`` or an ``IntoExprColumn``
            (``pl.Expr``, ``pl.Series`` or column name ``str``) carrying one bound per row.
        max: Inclusive upper bound, with ``max >= min`` (``min == max`` is a one-point mass). Same
            accepted types as ``min``.

    An invalid parameterisation (``max < min``, or a width ``max - min + 1`` overflowing ``Int64``)
    is not checked at construction; as everywhere, it raises ``InvalidOperation`` (a
    ``ComputeError``) when a method is evaluated. Null bounds propagate to null.
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
    def support_size(self) -> pl.Expr:
        """Support count ``N = max - min + 1``, as ``Float64``.

        Validated in Rust so every closed form raises on an invalid parameterisation (``max < min``,
        an overflowing width) consistently with the sampler; null bounds propagate. Above ``2**53``
        the count itself is rounded. Each mention is one validator pass over the bounds.
        """
        return self._checked_plugin_output("discreteuniform_range")

    @property
    def _checked_params(self) -> pl.Expr:
        """The validated support count; see `support_size`."""
        return self.support_size

    @property
    def _point_mass(self) -> pl.Expr:
        """``1 / support_size``, the mass on each support point.

        Read off the validator once so every consumer multiplies the same reciprocal: polars divides
        a column by a broadcast scalar through a different code path than column by column, and the
        two disagree in the last bit. Costs one or two ulp against an exact quotient.
        """
        return 1 / self.support_size

    @property
    def _min_f64(self) -> pl.Expr:
        """``min`` as ``Float64``, the operand the inverses clip against."""
        return self._min.cast(pl.Float64())

    @property
    def _max_f64(self) -> pl.Expr:
        """``max`` as ``Float64``, the operand the inverses clip against."""
        return self._max.cast(pl.Float64())

    @property
    def _midpoint(self) -> pl.Expr:
        """``(min + max) / 2``, formed as ``min + (max - min) // 2`` to stay in ``Int64``.

        ``min + max`` overflows well inside the range the validator accepts, and casting each bound to
        ``Float64`` first rounds both before they cancel (``1.2e-13`` relative for bounds straddling
        zero). The width is exact in ``Int64`` and ``min + width // 2`` lies in ``[min, max]``, so the
        cast is the only rounding.
        """
        width = self._max - self._min
        return (self._min + width // 2).cast(pl.Float64()) + (width % 2) * 0.5

    def _is_on_support(self, value: pl.Expr) -> pl.Expr:
        """Whether ``value`` is an integer inside ``[min, max]``, where the mass sits."""
        return (value >= self._min) & (value <= self._max) & (value.floor() == value)

    def _points_at_or_below(self, value: pl.Expr) -> pl.Expr:
        """``k``, the number of support points at or below ``value``.

        Only ever read where ``k`` lies in ``[1, N - 1]``, which is what keeps the subtraction in
        ``Int64`` where it is exact. Rows outside the support do evaluate it and may wrap; their
        branch discards the result.
        """
        return value.floor() - self._min + 1

    def _pmf(self, value: pl.Expr) -> pl.Expr:
        """``1 / N`` on the integer support, ``0`` off it, ``null`` for null bounds.

        The null branch is explicit because `_is_on_support` reads the bounds, so a null bound makes
        its condition null rather than false, and a bare ``otherwise(0.0)`` would report a confident
        "off the support" for a row whose support is unknown.
        """
        return (
            pl.when(self._is_on_support(value))
            .then(self._point_mass)
            .when(self.support_size.is_not_null())
            .then(0.0)
            .otherwise(pl.lit(None, dtype=pl.Float64()))
        )

    def _log_pmf(self, value: pl.Expr) -> pl.Expr:
        """``-log(N)`` on the integer support, ``-inf`` off it, ``null`` for null bounds.

        Overrides the base ``pmf().log()`` to skip the division: the mass is exactly ``1 / N``, so its
        log reads straight off the count. Same null branch as `_pmf`.
        """
        return (
            pl.when(self._is_on_support(value))
            .then(-self.support_size.log())
            .when(self.support_size.is_not_null())
            .then(float("-inf"))
            .otherwise(pl.lit(None, dtype=pl.Float64()))
        )

    def _cdf(self, value: pl.Expr) -> pl.Expr:
        """``k * (1 / N)`` inside the support, ``0`` below ``min``, ``1`` from ``max`` up.

        The explicit endpoints make ``cdf(max) == 1`` an answer rather than the limit of a clamp,
        which is the visible difference from scipy's exclusive upper bound: ``N * (1 / N)`` is not
        ``1.0`` for 483 of the first 4000 support counts. They also keep `_points_at_or_below` inside
        the support, where its subtraction is exact.
        """
        return (
            pl.when(self.support_size.is_not_null())
            .then(
                pl.when(value < self._min)
                .then(0.0)
                .when(value >= self._max)
                .then(1.0)
                .otherwise(self._points_at_or_below(value) * self._point_mass)
            )
            .otherwise(pl.lit(None, dtype=pl.Float64()))
        )

    def _sf(self, value: pl.Expr) -> pl.Expr:
        """``(max - floor(value)) * (1 / N)`` inside the support, ``1`` below ``min``, ``0`` from ``max`` up.

        A direct count of the support points above ``value``, mirroring `_cdf`. Overrides the base
        ``1 - cdf``, whose subtraction absorbs the whole tail once the cdf rounds towards ``1``. The
        chain sits behind a null gate because the ``1.0`` branch reads only ``min``.
        """
        return (
            pl.when(self.support_size.is_not_null())
            .then(
                pl.when(value < self._min)
                .then(1.0)
                .when(value >= self._max)
                .then(0.0)
                .otherwise((self._max - value.floor()) * self._point_mass)
            )
            .otherwise(pl.lit(None, dtype=pl.Float64()))
        )

    def _log_cdf(self, value: pl.Expr) -> pl.Expr:
        """``log(k / N)``, through ``log1p`` of the survival ratio once the cdf passes one half.

        Overrides the base ``cdf().log()``, whose relative error one support point below the top grows
        with ``N``: ``8.7e-16`` at ``N = 1e3`` but ``2.8e-08`` at ``N = 1e9``. Same branch
        `Uniform._log_cdf` takes. The null gate is there for the reason in `_sf`.
        """
        count = self._points_at_or_below(value)
        return (
            pl.when(self.support_size.is_not_null())
            .then(
                pl.when(value < self._min)
                .then(float("-inf"))
                .when(value >= self._max)
                .then(0.0)
                .when(count * 2.0 > self.support_size)
                .then((-((self.support_size - count) * self._point_mass)).log1p())
                .otherwise((count * self._point_mass).log())
            )
            .otherwise(pl.lit(None, dtype=pl.Float64()))
        )

    def _log_sf(self, value: pl.Expr) -> pl.Expr:
        """The mirror of `_log_cdf`: the near-certain side is the lower one, so ``log1p`` sits there.

        ``0`` below ``min`` and ``-inf`` at or above ``max``, and the same null gate.
        """
        count = self._points_at_or_below(value)
        return (
            pl.when(self.support_size.is_not_null())
            .then(
                pl.when(value < self._min)
                .then(0.0)
                .when(value >= self._max)
                .then(float("-inf"))
                .when(count * 2.0 > self.support_size)
                .then(((self.support_size - count) * self._point_mass).log())
                .otherwise((-(count * self._point_mass)).log1p())
            )
            .otherwise(pl.lit(None, dtype=pl.Float64()))
        )

    def _ppf(self, quantile: pl.Expr) -> pl.Expr:
        """``min + ceil(quantile * N) - 1``, corrected and clamped; null outside ``[0, 1]``.

        Matches scipy's ``randint.ppf`` including its correction step, which probes the point below and
        keeps it whenever that point's cdf already reaches ``quantile``, settling the step boundaries
        where an ulp of noise in ``quantile * N`` would skip a support point. The probe's ``>=`` has no
        slack, so it reads an honestly rounded column-by-column quotient; that is why the validator
        takes ``quantile`` as a `_checked_plugin_output` length input, which makes it run per row.

        Both inverses lose support points above ``2**53``; see docs/explanation/accuracy.md.
        """
        support_size_per_row = self._checked_plugin_output("discreteuniform_range", quantile)
        min_f64, max_f64 = self._min_f64, self._max_f64
        candidate = min_f64 + (quantile * support_size_per_row).ceil() - 1.0
        point_below = (candidate - 1.0).clip(min_f64, max_f64)
        return (
            pl.when(quantile.is_between(0, 1))
            .then(
                pl.when(self._points_at_or_below(point_below) / support_size_per_row >= quantile)
                .then(point_below)
                .otherwise(candidate)
                .clip(min_f64, max_f64)
            )
            .otherwise(pl.lit(None, dtype=pl.Float64()))
        )

    def _isf(self, quantile: pl.Expr) -> pl.Expr:
        """``max - floor(quantile * N)``, corrected and clamped; null outside ``[0, 1]``.

        The smallest support point whose survival mass is at most ``quantile``. Entered against
        ``quantile`` itself rather than the base ``ppf(1 - quantile)``: the complement rounds before the
        inverse runs, and the survival steps sit at multiples of ``1 / N``, exactly where ``1 - q`` has
        spent its precision.

        The correction mirrors `_ppf`'s but probes the neighbour **above**, the only direction it can
        need: a rounded ``quantile * N`` can only floor one too high.
        """
        support_size_per_row = self._checked_plugin_output("discreteuniform_range", quantile)
        min_f64, max_f64 = self._min_f64, self._max_f64
        candidate = max_f64 - (quantile * support_size_per_row).floor()
        return (
            pl.when(quantile.is_between(0, 1))
            .then(
                pl.when((max_f64 - candidate) / support_size_per_row <= quantile)
                .then(candidate)
                .otherwise(candidate + 1.0)
                .clip(min_f64, max_f64)
            )
            .otherwise(pl.lit(None, dtype=pl.Float64()))
        )

    def mean(self) -> pl.Expr:
        """Expected value, ``(min + max) / 2``. See `_midpoint` for how the sum avoids overflowing."""
        return self._moment(self._midpoint)

    def variance(self) -> pl.Expr:
        """Variance, ``(N**2 - 1) / 12``."""
        return (self.support_size**2 - 1) * _ONE_TWELFTH

    def median(self) -> pl.Expr:
        """Median, the midpoint ``(min + max) / 2``, which for an even support size is not a support point.

        Overrides the base ``ppf(0.5)``, and **diverges from scipy**, which reports that support point:
        ``scipy.stats.randint(low=1, high=7).median()`` is ``3.0`` against this crate's ``3.5``.
        """
        return self._moment(self._midpoint)

    def entropy(self) -> pl.Expr:
        """Shannon entropy in nats, ``log(N)``."""
        return self.support_size.log()
