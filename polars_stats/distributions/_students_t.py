from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar

import polars as pl

from polars_stats.distributions._base import (
    UNDEFINED_MOMENT,
    ContinuousDistribution,
    coerce_param,
    register_plugin,
    scalar_float,
    scalar_kwargs,
)

if TYPE_CHECKING:
    from polars_stats._typing import DistributionName, IntoExprColumn

_MEAN_DEFINED_ABOVE = 1
"""The largest `dof` with no mean."""

_VARIANCE_FINITE_ABOVE = 2
"""The largest `dof` whose variance diverges."""


class StudentsT(ContinuousDistribution):
    """Student's t distribution with ``dof`` degrees of freedom, location ``loc`` and scale ``scale``.

    Equivalent to ``scipy.stats.t(df=dof, loc=loc, scale=scale)``, with the standard ``loc=0.0``, ``scale=1.0`` as
    defaults: ``StudentsT(dof=5.0)``. ``statrs`` names the degrees of freedom ``freedom``. The ``dof -> inf`` limit is
    ``Normal(mu=loc, sigma=scale)``, which ``dof`` itself excludes.

    Arguments:
        dof: Degrees of freedom, with ``dof > 0`` and not necessarily an integer. Either a Python ``float`` or an
            ``IntoExprColumn`` (``pl.Expr``, ``pl.Series`` or column name ``str``) carrying one ``dof`` per row.
        loc: Location parameter, the median and the mode. Same accepted types as ``dof``.
        scale: Scale parameter, with ``scale > 0``. Same accepted types as ``dof``.

    ``mean()`` is **null** for ``dof <= 1``, where it has no value (as for ``Cauchy``, which is ``StudentsT(1.0)``).
    ``variance()`` and ``std()`` are **null** there too, and **``+inf``** for ``1 < dof <= 2``, where the integral
    diverges. scipy returns ``nan`` where these are null.

    An invalid parameterisation (``dof <= 0``, ``scale <= 0`` or a non-finite parameter, ``dof = inf`` included) is
    not checked at construction; it raises ``InvalidOperation`` (a ``ComputeError``) when any method is evaluated. A
    null parameter nulls every method.
    """

    _dof: pl.Expr
    _loc: pl.Expr
    _scale: pl.Expr
    _distribution_name: ClassVar[DistributionName] = "studentst"

    def __init__(
        self,
        dof: float | IntoExprColumn,
        loc: float | IntoExprColumn = 0.0,
        scale: float | IntoExprColumn = 1.0,
    ) -> None:
        self._dof = coerce_param(dof, name="dof")
        self._loc = coerce_param(loc, name="loc")
        self._scale = coerce_param(scale, name="scale")
        self._scalar_kwargs = scalar_kwargs(dof=scalar_float(dof), loc=scalar_float(loc), scale=scalar_float(scale))

    @property
    def _param_exprs(self) -> tuple[pl.Expr, ...]:
        return (self._dof, self._loc, self._scale)

    @property
    def _validated_params(self) -> pl.Expr:
        return self._validated("scale", self._scale)

    def mean(self) -> pl.Expr:
        """Expected value, ``loc`` for ``dof > 1``, else **null**."""
        return self._moment(pl.when(self._dof > _MEAN_DEFINED_ABOVE).then(self._loc).otherwise(UNDEFINED_MOMENT))

    def variance(self) -> pl.Expr:
        """Variance, ``scale**2 * dof / (dof - 2)`` for ``dof > 2``, ``+inf`` for ``1 < dof <= 2``, else **null**."""
        return self._moment(self._spread_by_dof(self._scale**2 * (self._dof / (self._dof - 2))))

    def std(self) -> pl.Expr:
        """Standard deviation, ``scale * sqrt(dof / (dof - 2))``, and ``+inf`` or **null** where ``variance`` is.

        Not ``variance().sqrt()``: squaring and unsquaring the scale saturates about 300 decades earlier.
        """
        return self._moment(self._spread_by_dof(self._scale * (self._dof / (self._dof - 2)).sqrt()))

    def median(self) -> pl.Expr:
        """Median, ``loc``."""
        return self._moment(self._loc)

    def entropy(self) -> pl.Expr:
        """Differential entropy, ``log(scale)`` plus the standard t's, which depends on ``dof`` alone.

        The standard entropy is ``(dof + 1) / 2 * (digamma((dof + 1) / 2) - digamma(dof / 2)) +
        log(sqrt(dof) * B(dof / 2, 1 / 2))``.
        """
        standard = register_plugin(self._distribution_name, "entropy", (self._dof,))
        return self._moment(self._scale.log() + standard)

    def _spread_by_dof(self, finite: pl.Expr) -> pl.Expr:
        """`finite` past the variance's divergence, `+inf` up to it, and null where the mean has no value."""
        return (
            pl.when(self._dof > _VARIANCE_FINITE_ABOVE)
            .then(finite)
            .when(self._dof > _MEAN_DEFINED_ABOVE)
            .then(pl.lit(math.inf))
            .otherwise(UNDEFINED_MOMENT)
        )
