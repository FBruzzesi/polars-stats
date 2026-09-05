"""A null `p` nulls every answer that depends on it, and only those.

`pmf(2) = 0`, `cdf(-1) = 0` and `sf(1) = 0` carry no `p`, so they survive a null one; the class
docstring promises it and `Geometric` behaves the same way.

The contract lives in how the Rust bodies derive their branches: `p` arrives as an `Option<f64>` and
only the slots whose answer reads it are `p.map(...)`. Threading `p` through one `Option` chain
around a whole body would null the constants out and leave the rest of the suite green, since the
per-method null tests all evaluate on the support.
"""

from __future__ import annotations

import math

import polars as pl
import pytest

from polars_stats import Bernoulli

_NEG_INF = float("-inf")

# (method, value, expected) for the answers that hold with no `p` at all.
_OFF_SUPPORT: list[tuple[str, float, float]] = [
    ("pmf", -1.0, 0.0),
    ("pmf", 0.5, 0.0),
    ("pmf", 2.0, 0.0),
    ("log_pmf", -1.0, _NEG_INF),
    ("log_pmf", 0.5, _NEG_INF),
    ("log_pmf", 2.0, _NEG_INF),
    ("cdf", -1.0, 0.0),
    ("cdf", 1.0, 1.0),
    ("log_cdf", -1.0, _NEG_INF),
    ("log_cdf", 1.0, 0.0),
    ("sf", -1.0, 1.0),
    ("sf", 1.0, 0.0),
    ("log_sf", -1.0, 0.0),
    ("log_sf", 1.0, _NEG_INF),
]

# (method, value) pairs whose answer reads `p`, so a null one must reach the result.
_ON_SUPPORT: list[tuple[str, float]] = [
    ("pmf", 0.0),
    ("pmf", 1.0),
    ("log_pmf", 0.0),
    ("log_pmf", 1.0),
    ("cdf", 0.0),
    ("log_cdf", 0.0),
    ("sf", 0.0),
    ("log_sf", 0.0),
]

_MOMENTS = ["mean", "variance", "std", "median", "entropy"]


def _null_p() -> pl.DataFrame:
    return pl.DataFrame({"p": [None]}, schema={"p": pl.Float64})


@pytest.mark.parametrize(
    ("method", "value", "expected"),
    _OFF_SUPPORT,
    ids=[f"{method}({value})" for method, value, _ in _OFF_SUPPORT],
)
def test_off_support_constant_survives_a_null_p(method: str, value: float, expected: float) -> None:
    result = _null_p().select(r=getattr(Bernoulli(p=pl.col("p")), method)(value))["r"].item()
    assert result == expected


@pytest.mark.parametrize(("method", "value"), _ON_SUPPORT, ids=[f"{method}({value})" for method, value in _ON_SUPPORT])
def test_on_support_value_nulls_under_a_null_p(method: str, value: float) -> None:
    assert _null_p().select(r=getattr(Bernoulli(p=pl.col("p")), method)(value))["r"].item() is None


@pytest.mark.parametrize("method", _MOMENTS)
def test_moment_nulls_under_a_null_p(method: str) -> None:
    assert _null_p().select(r=getattr(Bernoulli(p=pl.col("p")), method)())["r"].item() is None


@pytest.mark.parametrize("method", ["ppf", "isf"])
@pytest.mark.parametrize("quantile", [0.0, 0.5, 1.0, 2.0])
def test_inverse_nulls_under_a_null_p(method: str, quantile: float) -> None:
    """Null inside `[0, 1]` because the answer needs `p`, and null outside it by the domain contract.

    The endpoints are probed because `ppf(1.0)` reads its own derived slot rather than the cdf step.
    """
    assert _null_p().select(r=getattr(Bernoulli(p=pl.col("p")), method)(quantile))["r"].item() is None


def test_nan_value_stays_nan_under_a_null_p() -> None:
    """A `NaN` evaluation point short-circuits before the branches, so a null `p` does not null it.

    The one row where the local driver diverges from the shared `value_keyed_per_row`, which nulls a
    row on any null parameter. Reached through the private hook, since the public wrapper answers
    `NaN` on its own.
    """
    result = _null_p().select(r=Bernoulli(p=pl.col("p"))._pmf(pl.lit(math.nan)))["r"].item()
    assert math.isnan(result)
