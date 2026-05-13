from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl


# TODO(FBruzzesi): Investigate better implementations for log_* methods over
# the naive ones due to concerns on numerical stability


class _UnivariateDistribution(ABC):
    """Abstract base class for a univariate probability distribution.

    Subclasses model a parameterised distribution and expose its standard functional forms as polars expressions.
    Parameters may be Python scalars or `pl.Expr`, which lets a single instance describe a different distribution per
    row (e.g. `Normal(mu=pl.col("mu"), sigma=1.0)`).

    The interface mirrors `scipy.stats.rv_continuous` / `rv_discrete` but returns `pl.Expr` instead of NumPy arrays.
    """

    @abstractmethod
    def sample(self, seed: int | None = None) -> pl.Expr:
        """Draw one random variate per row.

        Returns a polars Expr evaluating to a column with one variate per input row.
        """

    @abstractmethod
    def samples(self, size: int, seed: int | None = None) -> pl.Expr:
        """Draw `size` random variates per row.

        Returns polars Expr evaluating to a column of `Array(inner=..., shape=size)`.
        """

    @abstractmethod
    def cdf(self, value: float | pl.Expr) -> pl.Expr:
        """Cumulative distribution function, `P(X <= value)`."""

    def log_cdf(self, value: float | pl.Expr) -> pl.Expr:
        """Natural logarithm of the cdf."""
        return self.cdf(value).log()

    def sf(self, value: float | pl.Expr) -> pl.Expr:
        """Survival function, `P(X > value) = 1 - cdf(value)`.

        Subclasses should override when a closed form gives better accuracy in the upper tail.
        """
        return 1 - self.cdf(value)

    def log_sf(self, value: float | pl.Expr) -> pl.Expr:
        """Natural logarithm of the survival function."""
        return self.sf(value).log()

    @abstractmethod
    def ppf(self, quantile: float | pl.Expr) -> pl.Expr:
        """Percent point function (inverse cdf).

        `quantile` must lie in `[0, 1]`. Nulls are propagated; any other out-of-range value raises an error.
        """

    def isf(self, quantile: float | pl.Expr) -> pl.Expr:
        """Inverse survival function, `ppf(1 - quantile)`."""
        return self.ppf(1 - quantile)

    @abstractmethod
    def mean(self) -> pl.Expr:
        """Expected value `E[X]`."""

    @abstractmethod
    def variance(self) -> pl.Expr:
        """Variance `Var[X] = E[(X - E[X])^2]`."""

    def std(self) -> pl.Expr:
        """Standard deviation, `sqrt(variance)`."""
        return self.variance().sqrt()

    def median(self) -> pl.Expr:
        """Median, `ppf(0.5)`."""
        return self.ppf(0.5)

    @abstractmethod
    def entropy(self) -> pl.Expr:
        """Differential or Shannon entropy, in nats."""


class DiscreteDistribution(_UnivariateDistribution, ABC):
    """Abstract base class for discrete univariate distributions."""

    @abstractmethod
    def pmf(self, value: float | pl.Expr) -> pl.Expr:
        """Probability mass function, `P(X = value)`."""

    def log_pmf(self, value: float | pl.Expr) -> pl.Expr:
        """Natural logarithm of the pmf."""
        return self.pmf(value).log()


class ContinuousDistribution(_UnivariateDistribution, ABC):
    """Abstract base class for continuous univariate distributions."""

    @abstractmethod
    def pdf(self, value: float | pl.Expr) -> pl.Expr:
        """Probability density function evaluated at `value`."""

    def log_pdf(self, value: float | pl.Expr) -> pl.Expr:
        """Natural logarithm of the pdf."""
        return self.pdf(value).log()
