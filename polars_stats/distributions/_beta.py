from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from polars_stats.distributions._base import ContinuousDistribution, coerce_param, scalar_float, scalar_kwargs

if TYPE_CHECKING:
    import polars as pl

    from polars_stats._typing import DistributionName, IntoExprColumn


class Beta(ContinuousDistribution):
    """Beta distribution on ``[0, 1]`` with shape parameters ``a`` (alpha) and ``b`` (beta).

    Equivalent to ``scipy.stats.beta(a, b)``. The parameter names follow scipy; ``statrs`` calls
    them ``shape_a`` / ``shape_b``.

    Arguments:
        a: First shape parameter (alpha), with ``a > 0``. Either a Python ``float`` or an ``IntoExprColumn``
            (``pl.Expr``, ``pl.Series`` or column name ``str``) carrying one shape per row.
        b: Second shape parameter (beta), with ``b > 0``. Same accepted types as ``a``.

    An invalid shape (``a <= 0``, ``b <= 0``, or a non-finite parameter) is not checked at construction; it
    raises ``InvalidOperation`` (a ``ComputeError``) when any method is evaluated. Null parameters propagate
    to null.

    The support is ``[0, 1]``: ``pdf`` is ``0`` outside it, and when a shape is ``< 1`` the density
    diverges (``inf`` or large finite values) at the corresponding boundary.
    """

    _a: pl.Expr
    _b: pl.Expr
    _distribution_name: ClassVar[DistributionName] = "beta"

    def __init__(self, a: float | IntoExprColumn, b: float | IntoExprColumn) -> None:
        self._a = coerce_param(a, name="a")
        self._b = coerce_param(b, name="b")
        self._scalar_kwargs = scalar_kwargs(a=scalar_float(a), b=scalar_float(b))

    @property
    def _param_exprs(self) -> tuple[pl.Expr, ...]:
        return (self._a, self._b)

    @property
    def _validated_params(self) -> pl.Expr:
        return self._validated("params", self._b)

    def _log_cdf(self, value: pl.Expr) -> pl.Expr:
        """No Rust ``ln_cdf`` body yet, so this underflows to ``-inf`` where ``cdf`` rounds to ``0``."""
        return self._cdf(value).log()

    def _log_sf(self, value: pl.Expr) -> pl.Expr:
        """No Rust ``ln_sf`` body yet, so this underflows to ``-inf`` where ``sf`` rounds to ``0``."""
        return self._sf(value).log()

    def mean(self) -> pl.Expr:
        """Expected value, ``a / (a + b)``."""
        return self._moment(self._a / (self._a + self._b))

    def variance(self) -> pl.Expr:
        """Variance, ``a * b / ((a + b)^2 * (a + b + 1))``."""
        return self._moment(self._a * self._b / ((self._a + self._b) ** 2 * (self._a + self._b + 1)))

    def entropy(self) -> pl.Expr:
        """Differential entropy in nats, ``ln B(a, b) - (a - 1) psi(a) - (b - 1) psi(b) + (a + b - 2) psi(a + b)``.

        Log-Beta and digamma have no elementary closed form, so ``beta_entropy`` evaluates it in Rust.
        """
        return self._param_plugin("entropy")
