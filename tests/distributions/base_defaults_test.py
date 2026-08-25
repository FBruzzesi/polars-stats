from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import polars as pl
import pytest

from polars_stats.distributions._base import ContinuousDistribution, DiscreteDistribution

if TYPE_CHECKING:
    from polars_stats.distributions._base import _UnivariateDistribution

# Every shipped distribution overrides `_sf`, `_log_pdf` and `_log_pmf` with a form that keeps
# precision in a tail, so the composing defaults have no caller in the library. They are still what a
# new distribution inherits until it needs the accurate form.

_MEDIAN_QUANTILE = 0.5


class _UnitUniform(ContinuousDistribution):
    """`Uniform(0, 1)` in plain Polars, implementing the abstract hooks and nothing else."""

    _plugin_prefix: ClassVar[str] = "test_unit_uniform"

    def __init__(self) -> None:
        self._scalar_kwargs = None

    @property
    def _param_exprs(self) -> tuple[pl.Expr, ...]:
        return ()

    def _pdf(self, value: pl.Expr) -> pl.Expr:
        return pl.when(value.is_between(0.0, 1.0)).then(1.0).otherwise(0.0)

    def _cdf(self, value: pl.Expr) -> pl.Expr:
        return value.clip(0.0, 1.0)

    def _ppf(self, quantile: pl.Expr) -> pl.Expr:
        return quantile

    def mean(self) -> pl.Expr:
        return pl.lit(0.5)

    def variance(self) -> pl.Expr:
        return pl.lit(1 / 12)

    def entropy(self) -> pl.Expr:
        return pl.lit(0.0)


class _FairCoin(DiscreteDistribution):
    """`Bernoulli(0.5)` in plain Polars, implementing the abstract hooks and nothing else."""

    _plugin_prefix: ClassVar[str] = "test_fair_coin"

    def __init__(self) -> None:
        self._scalar_kwargs = None

    @property
    def _param_exprs(self) -> tuple[pl.Expr, ...]:
        return ()

    def _pmf(self, value: pl.Expr) -> pl.Expr:
        return pl.when(value.is_in([0.0, 1.0])).then(0.5).otherwise(0.0)

    def _cdf(self, value: pl.Expr) -> pl.Expr:
        return pl.when(value < 0).then(0.0).when(value < 1).then(0.5).otherwise(1.0)

    def _ppf(self, quantile: pl.Expr) -> pl.Expr:
        return (quantile > _MEDIAN_QUANTILE).cast(pl.Float64)

    def mean(self) -> pl.Expr:
        return pl.lit(0.5)

    def variance(self) -> pl.Expr:
        return pl.lit(0.25)

    def entropy(self) -> pl.Expr:
        return pl.lit(math.log(2.0))


@dataclass(frozen=True)
class _Toy:
    """One minimal distribution and what to evaluate it on.

    Arguments:
        dist: The instance.
        density: `"pdf"` or `"pmf"`. The density methods live on the family subclass, so a test
            written across both families reaches them by name.
        grid: Evaluation points spanning the support and both sides outside it.
        quantile: An interior quantile, away from the median so `ppf` and `isf` disagree.
        moments: Expected value of every parameter-free method, at that quantile for the inverses.
    """

    dist: _UnivariateDistribution
    density: str
    grid: list[float]
    quantile: float
    moments: dict[str, float]


_TOYS = {
    "continuous": _Toy(
        dist=_UnitUniform(),
        density="pdf",
        grid=[-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5],
        quantile=0.25,
        moments={
            "ppf": 0.25,
            "isf": 0.75,
            "mean": 0.5,
            "variance": 1 / 12,
            "std": math.sqrt(1 / 12),
            "median": 0.5,
            "entropy": 0.0,
        },
    ),
    "discrete": _Toy(
        dist=_FairCoin(),
        density="pmf",
        grid=[-1.0, 0.0, 0.5, 1.0, 2.0],
        quantile=0.75,
        moments={
            "ppf": 1.0,
            "isf": 0.0,
            "mean": 0.5,
            "variance": 0.25,
            "std": 0.5,
            "median": 0.0,
            "entropy": math.log(2.0),
        },
    ),
}

pytestmark = pytest.mark.parametrize("toy", _TOYS.values(), ids=list(_TOYS))


def test_the_default_sf_is_the_complement_of_cdf(toy: _Toy) -> None:
    got = pl.DataFrame({"x": toy.grid}).select(cdf=toy.dist.cdf(pl.col("x")), sf=toy.dist.sf(pl.col("x")))

    assert (got["cdf"] + got["sf"]).to_list() == [1.0] * len(toy.grid)


def test_the_default_sf_keeps_the_null_and_nan_contract(toy: _Toy) -> None:
    # `sf` wraps the hook in `propagate_null_and_nan`, so the default inherits the contract. Only the
    # null and `NaN` rows are asserted here; the value is the test above.
    frame = pl.DataFrame({"x": [toy.quantile, None, float("nan")]}, schema={"x": pl.Float64})

    got = frame.select(r=toy.dist.sf(pl.col("x")))["r"].to_list()

    assert got[0] is not None
    assert got[1] is None
    assert math.isnan(got[2])


def test_the_default_log_density_is_the_log_of_the_density(toy: _Toy) -> None:
    got = pl.DataFrame({"x": toy.grid}).select(
        direct=getattr(toy.dist, f"log_{toy.density}")(pl.col("x")),
        composed=getattr(toy.dist, toy.density)(pl.col("x")),
    )

    for direct, composed in zip(got["direct"], got["composed"], strict=True):
        assert direct == (math.log(composed) if composed > 0 else -math.inf)


def test_the_minimal_surface_is_coherent(toy: _Toy) -> None:
    # The remaining defaults a distribution inherits for free: `_isf` as `ppf(1 - q)`, `std` as
    # `sqrt(variance)`, `median` as `ppf(0.5)`.
    dist = toy.dist
    got = pl.DataFrame({"q": [toy.quantile]}).select(
        ppf=dist.ppf(pl.col("q")),
        isf=dist.isf(pl.col("q")),
        mean=dist.mean(),
        variance=dist.variance(),
        std=dist.std(),
        median=dist.median(),
        entropy=dist.entropy(),
    )

    assert got.row(0, named=True) == pytest.approx(toy.moments)
    # No parameter crosses FFI, which is why the samplers are out of reach for these two classes.
    assert dist._param_exprs == ()
