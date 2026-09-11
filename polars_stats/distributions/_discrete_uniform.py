from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import polars as pl

from polars_stats.distributions._base import DiscreteDistribution, coerce_int, scalar_int, scalar_kwargs

if TYPE_CHECKING:
    from polars_stats._typing import DistributionName, IntoExprColumn

_ONE_TWELFTH = 1 / 12
"""Multiplied by rather than divided against: polars spells ``column / 12`` as a division up to eight rows and
as a reciprocal multiply above, which would make ``variance`` row-count dependent."""


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
    is not checked at construction; it raises ``InvalidOperation`` (a ``ComputeError``) when a method is
    evaluated. Null bounds propagate to null.
    """

    _min: pl.Expr
    _max: pl.Expr
    _distribution_name: ClassVar[DistributionName] = "discreteuniform"

    def __init__(self, min: int | IntoExprColumn, max: int | IntoExprColumn) -> None:  # noqa: A002
        self._min = coerce_int(min, name="min")
        self._max = coerce_int(max, name="max")
        self._scalar_kwargs = scalar_kwargs(min=scalar_int(min), max=scalar_int(max))

    @property
    def _param_exprs(self) -> tuple[pl.Expr, ...]:
        return (self._min, self._max)

    @property
    def support_size(self) -> pl.Expr:
        """Support count ``N = max - min + 1``, as ``Float64``, validated in Rust; rounded above ``2**53``."""
        return self._param_plugin("range")

    @property
    def _validated_params(self) -> pl.Expr:
        return self.support_size

    @property
    def _midpoint(self) -> pl.Expr:
        """``(min + max) / 2`` as ``min + (max - min) // 2`` in ``Int64``, plus half if the width is odd.

        ``min + max`` overflows well inside the range the validator accepts, and casting each bound to
        ``Float64`` first rounds both before they cancel. The bounds are widened non-strictly so an
        out-of-range ``UInt64`` value nulls here and stays the validator's error to report.
        """
        lo = self._min.cast(pl.Int64(), strict=False)
        width = self._max.cast(pl.Int64(), strict=False) - lo
        return (lo + width // 2).cast(pl.Float64()) + (width % 2) * 0.5

    def mean(self) -> pl.Expr:
        """Expected value, ``(min + max) / 2``."""
        return self._moment(self._midpoint)

    def variance(self) -> pl.Expr:
        """Variance, ``(N**2 - 1) / 12``."""
        return (self.support_size**2 - 1) * _ONE_TWELFTH

    def median(self) -> pl.Expr:
        """Median, the midpoint ``(min + max) / 2``, which for an even support size is not a support point.

        **Diverges from scipy**, which reports that support point: ``scipy.stats.randint(low=1, high=7).median()``
        is ``3.0`` against this library's ``3.5``.
        """
        return self._moment(self._midpoint)

    def entropy(self) -> pl.Expr:
        """Shannon entropy in nats, ``log(N)``."""
        return self.support_size.log()
