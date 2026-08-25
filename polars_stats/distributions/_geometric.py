from __future__ import annotations

from typing import ClassVar

import polars as pl

from polars_stats._typing import Value
from polars_stats.distributions._base import DiscreteDistribution, coerce_param, scalar_kwargs


class Geometric(DiscreteDistribution):
    """Geometric distribution: trials until the first success.

    The support is `{1, 2, 3, ...}` (scipy's `geom` convention). Many textbooks define the
    geometric as failures before the first success, starting at 0; this class does not.

    Args:
        p: Success probability, in `(0, 1]`.
    """

    _plugin_prefix: ClassVar[str] = "geometric"

    def __init__(self, p: Value) -> None:
        self.p = coerce_param("p", p)
        self._param_exprs = (self.p,)
        self._scalar_kwargs = scalar_kwargs(p=p)

    def _checked_p(self) -> pl.Expr:
        return self._value_plugin("geometric_p")

    def _pmf(self, value: pl.Expr) -> pl.Expr:
        p = self._checked_p()
        k = value
        support = (k >= 1) & (k.floor() == k)
        return pl.when(support).then(((1 - p) ** (k - 1)) * p).otherwise(0.0)

    def _log_pmf(self, value: pl.Expr) -> pl.Expr:
        p = self._checked_p()
        k = value
        support = (k >= 1) & (k.floor() == k)
        return pl.when(support).then((k - 1) * (-p).log1p() + p.log()).otherwise(float("-inf"))

    def _cdf(self, value: pl.Expr) -> pl.Expr:
        p = self._checked_p()
        k = value
        return pl.when(k >= 1).then(1 - (1 - p) ** k.floor()).otherwise(0.0)

    def _sf(self, value: pl.Expr) -> pl.Expr:
        p = self._checked_p()
        k = value
        return pl.when(k >= 1).then((1 - p) ** k.floor()).otherwise(1.0)

    def _log_sf(self, value: pl.Expr) -> pl.Expr:
        p = self._checked_p()
        k = value
        return pl.when(k >= 1).then(k.floor() * (-p).log1p()).otherwise(0.0)

    def _log_cdf(self, value: pl.Expr) -> pl.Expr:
        p = self._checked_p()
        k = value
        sf = (1 - p) ** k.floor()
        return pl.when(k >= 1).then((-sf).log1p()).otherwise(float("-inf"))

    def _ppf(self, value: pl.Expr) -> pl.Expr:
        p = self._checked_p()
        q = value
        valid = ((q >= 0) & (q <= 1)).fill_null(False)
        p_is_one = (p == 1.0).fill_null(False)
        formula = ((1 - q).log() / (-p).log1p()).ceil()
        degenerate = pl.when(q == 0.0).then(0.0).otherwise(1.0)
        return pl.when(~valid).then(None).when(p_is_one).then(degenerate).otherwise(formula)

    def _mean(self) -> pl.Expr:
        return 1 / self._checked_p()

    def _variance(self) -> pl.Expr:
        p = self._checked_p()
        return (1 - p) / (p * p)

    def _entropy(self) -> pl.Expr:
        p = self._checked_p()
        p_is_one = (p == 1.0).fill_null(False)
        formula = (-(1 - p) * (-p).log1p() - p * p.log()) / p
        return pl.when(p_is_one).then(0.0).otherwise(formula)
