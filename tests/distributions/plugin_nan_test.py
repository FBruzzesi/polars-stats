"""Every value-keyed method propagates a `NaN` evaluation point as `NaN`, `ppf` / `isf` included.

The `NaN` short-circuit lives in the shared drivers in `src/distributions/mod.rs` (`value_keyed_scalar`
for constant parameters, `value_keyed_binary` and `value_keyed_ternary` for column parameters) and is
load-bearing in two places:

* `Beta` `cdf` / `sf`: statrs' regularized incomplete beta panics on a `NaN` evaluation point, so
  without the short-circuit the whole query aborts.
* `Binomial`: `NaN < 0.0` is false and `NaN.floor() as u64` saturates to `0`, so unguarded bodies
  would return a confident `P(X <= 0)` (and a `pmf` of `0.0`).

Each driver repeats the short-circuit rather than sharing one, so every case here runs against all
three and the drivers cannot drift apart. Both plugin shapes are covered: scalar-parameter instances
route to the `<method>_scalar` twins, column-parameter instances to the per-row plugins.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import polars as pl
import pytest

from polars_stats import Bernoulli, Beta, Binomial, DiscreteUniform, Exponential, Geometric, LogNormal, Normal, Uniform
from polars_stats.distributions._base import DiscreteDistribution

if TYPE_CHECKING:
    from polars_stats._typing import PolarsDataType
    from polars_stats.distributions._base import _UnivariateDistribution


def _col(value: float, dtype: PolarsDataType | None = None) -> pl.Expr:
    """A full-length constant column expr (`pl.repeat`), forcing the per-row plugin path."""
    return pl.repeat(value, n=pl.len(), dtype=dtype)


_CASES = [
    ("bernoulli-scalar", Bernoulli(p=0.3)),
    ("bernoulli-columns", Bernoulli(p=_col(0.3))),
    ("normal-scalar", Normal(mu=0.5, sigma=2.0)),
    ("normal-columns", Normal(mu=_col(0.5), sigma=_col(2.0))),
    ("lognormal-scalar", LogNormal(mu=0.5, sigma=2.0)),
    ("lognormal-columns", LogNormal(mu=_col(0.5), sigma=_col(2.0))),
    ("beta-scalar", Beta(a=2.0, b=3.0)),
    ("beta-columns", Beta(a=_col(2.0), b=_col(3.0))),
    ("binomial-scalar", Binomial(n=5, p=0.4)),
    ("binomial-columns", Binomial(n=_col(5, pl.Int64()), p=_col(0.4))),
    ("exponential-scalar", Exponential(rate=1.5)),
    ("exponential-columns", Exponential(rate=_col(1.5))),
    ("geometric-scalar", Geometric(p=0.3)),
    ("geometric-columns", Geometric(p=_col(0.3))),
    ("discreteuniform-scalar", DiscreteUniform(min=-2, max=9)),
    ("discreteuniform-columns", DiscreteUniform(min=_col(-2, pl.Int64()), max=_col(9, pl.Int64()))),
    ("uniform-scalar", Uniform(min=-2.0, max=9.0)),
    ("uniform-columns", Uniform(min=_col(-2.0), max=_col(9.0))),
]


def _value_keyed_methods(dist: _UnivariateDistribution) -> tuple[str, ...]:
    """Every value-keyed method of `dist`, density methods named by family."""
    density = ("pmf", "log_pmf") if isinstance(dist, DiscreteDistribution) else ("pdf", "log_pdf")
    return (*density, "cdf", "log_cdf", "sf", "log_sf", "ppf", "isf")


@pytest.mark.parametrize(("label", "dist"), _CASES, ids=[label for label, _ in _CASES])
def test_value_keyed_method_propagates_nan(label: str, dist: _UnivariateDistribution) -> None:
    """Each method yields `NaN` (not null, not a confident constant) for a `NaN` evaluation point."""
    frame = pl.DataFrame({"x": [float("nan")]})
    for method in _value_keyed_methods(dist):
        out = frame.select(r=getattr(dist, method)(pl.col("x")))["r"]
        assert out.null_count() == 0, f"{label}.{method} nulled a NaN evaluation point"
        assert math.isnan(out.item()), f"{label}.{method} did not propagate NaN"
