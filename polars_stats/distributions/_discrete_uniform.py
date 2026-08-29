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
        min: Inclusive lower bound, an integer anywhere in ``Int64``: ``[-2**63, 2**63 - 1]``.
            Either a Python ``int`` (rejected at construction outside that range) or an
            ``IntoExprColumn`` (``pl.Expr``, ``pl.Series`` or column name ``str``) carrying one
            bound per row; a column may be any integer dtype, judged by its values fitting
            ``Int64``.
        max: Inclusive upper bound, with ``max >= min`` (``min == max`` is a one-point mass) and
            the width ``max - min + 1`` fitting ``Int64``. Same accepted types and range as
            ``min``.

    An invalid parameterisation (``max < min``, or a width ``max - min + 1`` overflowing ``Int64``)
    is not checked at construction; as everywhere, it raises ``InvalidOperation`` (a
    ``ComputeError``) when a method is evaluated. Null bounds propagate to null.

    The closed forms run in Rust (``src/distributions/discrete_uniform.rs``), one plugin pass per
    method, where an integer evaluation point keeps exact integer arithmetic; the numeric notes
    (endpoint contract, ``log1p`` cut-over, inverse correction) live on the Rust bodies.
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

        Validated in Rust so every moment raises on an invalid parameterisation (``max < min``, an
        overflowing width) consistently with the value-keyed methods and the sampler; null bounds
        propagate. Above ``2**53`` the count itself is rounded. Each mention is one validator pass
        over the bounds.
        """
        return self._checked_plugin_output("discreteuniform_range")

    @property
    def _checked_params(self) -> pl.Expr:
        """The validated support count; see `support_size`."""
        return self.support_size

    @property
    def _min_i64(self) -> pl.Expr:
        """``min`` widened to ``Int64``, the dtype `_midpoint`'s integer arithmetic assumes.

        A bound column keeps its own dtype until Rust widens it, so arithmetic on the raw bound runs
        in that dtype and wraps once the support outgrows it (``Int8`` bounds ``(-100, 100)``).
        Non-strict, so an out-of-range ``UInt64`` value nulls here and stays the validator's error
        to report.
        """
        return self._min.cast(pl.Int64(), strict=False)

    @property
    def _max_i64(self) -> pl.Expr:
        """``max`` widened to ``Int64``; see `_min_i64`."""
        return self._max.cast(pl.Int64(), strict=False)

    @property
    def _midpoint(self) -> pl.Expr:
        """``(min + max) / 2``, formed as ``min + (max - min) // 2`` to stay in ``Int64``.

        ``min + max`` overflows well inside the range the validator accepts, and casting each bound to
        ``Float64`` first rounds both before they cancel (``1.2e-13`` relative for bounds straddling
        zero). The width is exact in ``Int64`` and ``min + width // 2`` lies in ``[min, max]``, so the
        cast is the only rounding.
        """
        width = self._max_i64 - self._min_i64
        return (self._min_i64 + width // 2).cast(pl.Float64()) + (width % 2) * 0.5

    def _pmf(self, value: pl.Expr) -> pl.Expr:
        """``1 / N`` on the integer support, ``0`` off it."""
        return self._value_plugin("discreteuniform_pmf", value)

    def _log_pmf(self, value: pl.Expr) -> pl.Expr:
        """``-log(N)`` on the integer support, ``-inf`` off it: the log reads straight off the count."""
        return self._value_plugin("discreteuniform_ln_pmf", value)

    def _cdf(self, value: pl.Expr) -> pl.Expr:
        """``k / N`` inside the support, with explicit endpoints so ``cdf(max) == 1`` exactly."""
        return self._value_plugin("discreteuniform_cdf", value)

    def _sf(self, value: pl.Expr) -> pl.Expr:
        """A direct count of the points above ``value``, not ``1 - cdf``, which absorbs the tail."""
        return self._value_plugin("discreteuniform_sf", value)

    def _log_cdf(self, value: pl.Expr) -> pl.Expr:
        """``log(k / N)``, through ``log1p`` of the survival ratio once the cdf passes one half."""
        return self._value_plugin("discreteuniform_ln_cdf", value)

    def _log_sf(self, value: pl.Expr) -> pl.Expr:
        """The mirror of `_log_cdf`: the near-certain side is the lower one, so ``log1p`` sits there."""
        return self._value_plugin("discreteuniform_ln_sf", value)

    def _ppf(self, quantile: pl.Expr) -> pl.Expr:
        """``min + ceil(quantile * N) - 1``, corrected at the step boundaries as scipy's ``randint.ppf`` is."""
        return self._value_plugin("discreteuniform_ppf", quantile)

    def _isf(self, quantile: pl.Expr) -> pl.Expr:
        """The smallest support point whose survival mass is at most ``quantile``.

        Entered against ``quantile`` itself rather than the base ``ppf(1 - quantile)``: the
        complement rounds before the inverse runs, and the survival steps sit at multiples of
        ``1 / N``, exactly where ``1 - q`` has spent its precision.
        """
        return self._value_plugin("discreteuniform_isf", quantile)

    def mean(self) -> pl.Expr:
        """Expected value, ``(min + max) / 2``. See `_midpoint` for how the sum avoids overflowing."""
        return self._moment(self._midpoint)

    def variance(self) -> pl.Expr:
        """Variance, ``(N**2 - 1) / 12``."""
        return (self.support_size**2 - 1) * _ONE_TWELFTH

    def median(self) -> pl.Expr:
        """Median, the midpoint ``(min + max) / 2``, which for an even support size is not a support point.

        Overrides the base ``ppf(0.5)``, and **diverges from scipy**, which reports that support point:
        ``scipy.stats.randint(low=1, high=7).median()`` is ``3.0`` against this library's ``3.5``.
        """
        return self._moment(self._midpoint)

    def entropy(self) -> pl.Expr:
        """Shannon entropy in nats, ``log(N)``."""
        return self.support_size.log()
