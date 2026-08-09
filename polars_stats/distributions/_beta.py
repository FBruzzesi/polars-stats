from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from polars_stats.distributions._base import (
    ContinuousDistribution,
    coerce_param,
    register_plugin,
    scalar_float,
    scalar_kwargs,
)

if TYPE_CHECKING:
    import polars as pl

    from polars_stats._typing import IntoExprColumn


class Beta(ContinuousDistribution):
    """Beta distribution on ``[0, 1]`` with shape parameters ``a`` (alpha) and ``b`` (beta).

    Equivalent to ``scipy.stats.beta(a, b)``. The parameter names follow scipy; ``statrs`` calls
    them ``shape_a`` / ``shape_b``.

    Arguments:
        a: First shape parameter (alpha), with ``a > 0``. Either a Python ``float`` or an ``IntoExprColumn``
            (``pl.Expr``, ``pl.Series`` or column name ``str``) carrying one shape per row.
        b: Second shape parameter (beta), with ``b > 0``. Same accepted types as ``a``.

    An invalid shape (``a <= 0``, ``b <= 0``, or a non-finite parameter) is not checked at construction; matching every
    other distribution, it raises ``InvalidOperation`` (a ``ComputeError``) when any method is evaluated.
    Null parameters propagate to null.

    The support is ``[0, 1]``: ``pdf`` is ``0`` outside it, and when a shape is ``< 1`` the density
    diverges to ``inf`` at the corresponding boundary (as does ``log_pdf``), matching scipy.
    """

    _a: pl.Expr
    _b: pl.Expr
    _plugin_prefix: ClassVar[str] = "beta"

    def __init__(self, a: float | IntoExprColumn, b: float | IntoExprColumn) -> None:
        self._a = coerce_param(a, name="a")
        self._b = coerce_param(b, name="b")
        self._scalar_kwargs = scalar_kwargs(a=scalar_float(a), b=scalar_float(b))

    @property
    def _param_exprs(self) -> tuple[pl.Expr, ...]:
        return (self._a, self._b)

    @property
    def _checked_params(self) -> pl.Expr:
        """``b`` validated in Rust against the full ``(a, b)`` parameterisation (raises otherwise).

        See ``_UnivariateDistribution._checked_params`` / ``_checked`` for the moment-gating contract.
        """
        return self._checked("beta_params", self._b)

    def _pdf(self, value: pl.Expr) -> pl.Expr:
        """Density via native ``statrs`` ``Continuous::pdf`` (Beta function); ``0`` outside ``[0, 1]``."""
        return self._value_plugin("beta_pdf", value)

    def _log_pdf(self, value: pl.Expr) -> pl.Expr:
        """Log-density via native ``Continuous::ln_pdf`` (more accurate than ``pdf().log()``)."""
        return self._value_plugin("beta_ln_pdf", value)

    def _cdf(self, value: pl.Expr) -> pl.Expr:
        """Cumulative distribution via native ``ContinuousCDF::cdf`` (regularized incomplete beta)."""
        return self._value_plugin("beta_cdf", value)

    def _sf(self, value: pl.Expr) -> pl.Expr:
        """Survival function via native ``ContinuousCDF::sf`` (accurate in the upper tail).

        ``isf`` inherits the base-class default ``ppf(1 - quantile)``.
        """
        return self._value_plugin("beta_sf", value)

    def _log_cdf(self, value: pl.Expr) -> pl.Expr:
        """Stable log-cdf via the Rust log-space incomplete-beta port.

        Overrides the naive ``cdf().log()``, which underflows to ``-inf`` once ``value**a`` rounds
        to ``0`` in the left corner (large ``a``, small ``value``), exactly the regime tail scoring
        needs finite. Beyond scipy there: ``scipy.stats.beta.logcdf`` is itself naive and returns
        ``-inf``.
        """
        return self._value_plugin("beta_ln_cdf", value)

    def _log_sf(self, value: pl.Expr) -> pl.Expr:
        """Stable log-sf via the Rust log-space incomplete-beta port.

        Overrides the naive ``sf().log()``, which underflows to ``-inf`` in the right corner
        (large ``b``, ``value`` near ``1``); mirror of ``_log_cdf``, and likewise finite where
        ``scipy.stats.beta.logsf`` is ``-inf``.
        """
        return self._value_plugin("beta_ln_sf", value)

    def _ppf(self, quantile: pl.Expr) -> pl.Expr:
        """Inverse cdf, by a bounded solve on the log-space incomplete beta (not statrs' AS 64).

        A quantile outside ``[0, 1]`` yields null; the endpoints map to the support bounds
        (``ppf(0) = 0``, ``ppf(1) = 1``), matching scipy. ``median`` is ``ppf(0.5)`` (the base-class default);
        the beta median has no closed form.
        """
        return self._value_plugin("beta_ppf", quantile)

    def _isf(self, quantile: pl.Expr) -> pl.Expr:
        """Inverse survival function, the same solve entered from the upper tail.

        Overrides the base-class default ``ppf(1 - quantile)``, which would take the branch that has
        to recover the mass as ``1 - (1 - quantile)``. The endpoints map to the support bounds in the
        opposite order to ``ppf`` (``isf(0) = 1``, ``isf(1) = 0``).
        """
        return self._value_plugin("beta_isf", quantile)

    def mean(self) -> pl.Expr:
        """Expected value, ``a / (a + b)``."""
        return self._moment(self._a / (self._a + self._b))

    def variance(self) -> pl.Expr:
        """Variance, ``a * b / ((a + b)^2 * (a + b + 1))``."""
        return self._moment(self._a * self._b / ((self._a + self._b) ** 2 * (self._a + self._b + 1)))

    def entropy(self) -> pl.Expr:
        """Differential entropy in nats, ``ln B(a, b) - (a - 1) psi(a) - (b - 1) psi(b) + (a + b - 2) psi(a + b)``."""
        # Unlike ``mean`` / ``variance`` there is no elementary closed form (log-Beta and digamma), so the
        # formula is evaluated by ``beta_entropy`` in Rust. For column parameters that runs once per row; for
        # scalar parameters it is computed **once** on length-1 inputs and broadcast to length-n behind the
        # ``_moment`` validity gate, so a constant's entropy is not re-evaluated on every row.
        if self._scalar_kwargs is None:
            return register_plugin("beta_entropy", (self._a, self._b))
        return self._moment(register_plugin("beta_entropy", self._scalar_lit_args()))
