from __future__ import annotations

import math
from typing import TYPE_CHECKING

import polars as pl
import pytest

from polars_stats import Exponential

if TYPE_CHECKING:
    from collections.abc import Callable

# Every closed-form method must propagate a null rate to a null result. The value-keyed methods are
# evaluated at an on-support point (`x >= 0`, `q in [0, 1]`) where the rate enters the formula, so the
# null flows through; below the support they return the rate-independent support constant instead (see
# `test_value_keyed_below_support_ignores_null_rate`).
_METHODS: dict[str, Callable[[Exponential], pl.Expr]] = {
    "pdf": lambda e: e.pdf(pl.lit(0.5)),
    "log_pdf": lambda e: e.log_pdf(pl.lit(0.5)),
    "cdf": lambda e: e.cdf(pl.lit(0.5)),
    "log_cdf": lambda e: e.log_cdf(pl.lit(0.5)),
    "sf": lambda e: e.sf(pl.lit(0.5)),
    "log_sf": lambda e: e.log_sf(pl.lit(0.5)),
    "ppf": lambda e: e.ppf(pl.lit(0.5)),
    "isf": lambda e: e.isf(pl.lit(0.5)),
    "mean": lambda e: e.mean(),
    "variance": lambda e: e.variance(),
    "std": lambda e: e.std(),
    "median": lambda e: e.median(),
    "entropy": lambda e: e.entropy(),
}


@pytest.mark.parametrize("expr_fn", _METHODS.values(), ids=list(_METHODS))
def test_method_propagates_null_in_rate(expr_fn: Callable[[Exponential], pl.Expr]) -> None:
    df = pl.DataFrame({"rate": [1.0, None, 2.0]}, schema={"rate": pl.Float64})
    result = df.select(r=expr_fn(Exponential(rate=pl.col("rate"))))["r"]
    assert result.is_null().to_list() == [False, True, False]


def test_value_keyed_below_support_ignores_null_rate() -> None:
    # Closed-form trait shared with Uniform: below the support all six value-keyed methods return the
    # rate-independent support constant, so a null rate does NOT null the row there. statrs-backed
    # distributions (Binomial) null it instead. Bernoulli keeps the same contract in Rust
    # (tests/distributions/bernoulli/null_param_test.py).
    df = pl.DataFrame({"rate": [1.0, None, 2.0]}, schema={"rate": pl.Float64})
    e = Exponential(rate=pl.col("rate"))
    assert df.select(r=e.pdf(pl.lit(-1.0)))["r"].to_list() == [0.0, 0.0, 0.0]
    assert df.select(r=e.cdf(pl.lit(-1.0)))["r"].to_list() == [0.0, 0.0, 0.0]
    assert df.select(r=e.sf(pl.lit(-1.0)))["r"].to_list() == [1.0, 1.0, 1.0]
    assert df.select(r=e.log_pdf(pl.lit(-1.0)))["r"].to_list() == [-math.inf] * 3
    assert df.select(r=e.log_cdf(pl.lit(-1.0)))["r"].to_list() == [-math.inf] * 3
    assert df.select(r=e.log_sf(pl.lit(-1.0)))["r"].to_list() == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("method", ["ppf", "isf"])
@pytest.mark.parametrize("quantile", [0.0, 0.5, 1.0, 2.0])
def test_inverse_nulls_under_a_null_rate(method: str, quantile: float) -> None:
    """Null inside `[0, 1]` because the answer needs the rate, and null outside it by the domain contract.

    Endpoints as well as interior: unlike the six value-keyed methods, neither inverse has a
    rate-free branch anywhere in its domain, so no `q` survives a null rate.
    """
    df = pl.DataFrame({"rate": [None]}, schema={"rate": pl.Float64})
    assert df.select(r=getattr(Exponential(rate=pl.col("rate")), method)(quantile))["r"].item() is None


def test_nan_value_stays_nan_under_a_null_rate() -> None:
    """A `NaN` evaluation point short-circuits before the branches, so a null rate does not null it.

    Reached through the private hook, since the public wrapper answers `NaN` on its own. Only the
    hook can see the short-circuit, and only there does it matter: polars orders `NaN` above every
    number, so a `value >= 0` predicate would put `NaN` on the support and let a null rate null it.
    """
    df = pl.DataFrame({"rate": [None]}, schema={"rate": pl.Float64})
    result = df.select(r=Exponential(rate=pl.col("rate"))._pdf(pl.lit(math.nan)))["r"].item()
    assert math.isnan(result)
