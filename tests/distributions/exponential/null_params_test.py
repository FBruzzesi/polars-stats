"""A null `rate` nulls every answer on the support, where the answer reads the rate directly.

Off-support, `NaN` and null evaluation points live in `null_nan_contract_test.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from polars_stats import Exponential

if TYPE_CHECKING:
    from collections.abc import Callable

# The value-keyed methods are evaluated at an on-support point (`x >= 0`, `q in [0, 1]`).
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


@pytest.mark.parametrize("method", ["ppf", "isf"])
@pytest.mark.parametrize("quantile", [0.0, 0.5, 1.0, 2.0])
def test_inverse_nulls_under_a_null_rate(method: str, quantile: float) -> None:
    df = pl.DataFrame({"rate": [None]}, schema={"rate": pl.Float64})
    assert df.select(r=getattr(Exponential(rate=pl.col("rate")), method)(quantile))["r"].item() is None
