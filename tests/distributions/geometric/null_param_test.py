"""A null `p` nulls every answer on the support, where the answer reads `p` directly.

Off-support, `NaN` and null evaluation points live in `null_nan_contract_test.py`.
"""

from __future__ import annotations

import polars as pl
import pytest

from polars_stats import Geometric

# (method, value) pairs whose answer reads `p`; `cdf(2.5)` is `cdf(2)`, so a non-integral point in
# range is on the support for the tail methods.
_ON_SUPPORT: list[tuple[str, float]] = [
    ("pmf", 1.0),
    ("pmf", 3.0),
    ("log_pmf", 1.0),
    ("log_pmf", 3.0),
    ("cdf", 1.0),
    ("cdf", 2.5),
    ("log_cdf", 1.0),
    ("log_cdf", 2.5),
    ("sf", 1.0),
    ("sf", 2.5),
    ("log_sf", 1.0),
    ("log_sf", 2.5),
]

_PARAMETER_ONLY = ["mean", "variance", "std", "median", "entropy"]


def _null_p() -> pl.DataFrame:
    return pl.DataFrame({"p": [None]}, schema={"p": pl.Float64})


@pytest.mark.parametrize(("method", "value"), _ON_SUPPORT, ids=[f"{method}({value})" for method, value in _ON_SUPPORT])
def test_on_support_value_nulls_under_a_null_p(method: str, value: float) -> None:
    assert _null_p().select(r=getattr(Geometric(p=pl.col("p")), method)(value))["r"].item() is None


@pytest.mark.parametrize("method", _PARAMETER_ONLY)
def test_moment_nulls_under_a_null_p(method: str) -> None:
    assert _null_p().select(r=getattr(Geometric(p=pl.col("p")), method)())["r"].item() is None


@pytest.mark.parametrize("method", ["ppf", "isf"])
@pytest.mark.parametrize("quantile", [0.0, 0.5, 1.0, 2.0])
def test_inverse_nulls_under_a_null_p(method: str, quantile: float) -> None:
    """Null inside `[0, 1]` because the answer needs `p`, and null outside it by the domain contract."""
    assert _null_p().select(r=getattr(Geometric(p=pl.col("p")), method)(quantile))["r"].item() is None
