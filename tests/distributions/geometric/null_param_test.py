"""A null `p` nulls every answer that depends on it, and only those.

`pmf(0) = 0`, `cdf(0) = 0` and `sf(0) = 1` are off-support constants with no `p` in them, so they
survive a null one; the class docstring promises it and `Bernoulli` behaves the same way. The split is
per method: a non-integral point in range carries no mass, so `pmf(2.5) = 0` survives, but `cdf(2.5)`
is `cdf(2)`, needs `p`, and nulls.

Nothing else pins the constants: the per-method null tests all evaluate on the support.
"""

from __future__ import annotations

import math

import polars as pl
import pytest

from polars_stats import Geometric

_NEG_INF = float("-inf")

# (method, value, expected) for the answers that hold with no `p` at all: below the support for every
# method, and at a non-integral point in range for the two mass methods.
_OFF_SUPPORT: list[tuple[str, float, float]] = [
    ("pmf", -1.0, 0.0),
    ("pmf", 0.0, 0.0),
    ("pmf", 0.5, 0.0),
    ("pmf", 2.5, 0.0),
    ("log_pmf", -1.0, _NEG_INF),
    ("log_pmf", 0.0, _NEG_INF),
    ("log_pmf", 0.5, _NEG_INF),
    ("log_pmf", 2.5, _NEG_INF),
    ("cdf", -1.0, 0.0),
    ("cdf", 0.0, 0.0),
    ("cdf", 0.5, 0.0),
    ("log_cdf", -1.0, _NEG_INF),
    ("log_cdf", 0.0, _NEG_INF),
    ("log_cdf", 0.5, _NEG_INF),
    ("sf", -1.0, 1.0),
    ("sf", 0.0, 1.0),
    ("sf", 0.5, 1.0),
    ("log_sf", -1.0, 0.0),
    ("log_sf", 0.0, 0.0),
    ("log_sf", 0.5, 0.0),
]

# (method, value) pairs whose answer reads `p`, so a null one must reach the result.
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


@pytest.mark.parametrize(
    ("method", "value", "expected"),
    _OFF_SUPPORT,
    ids=[f"{method}({value})" for method, value, _ in _OFF_SUPPORT],
)
def test_off_support_constant_survives_a_null_p(method: str, value: float, expected: float) -> None:
    result = _null_p().select(r=getattr(Geometric(p=pl.col("p")), method)(value))["r"].item()
    assert result == expected


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


def test_nan_value_stays_nan_under_a_null_p() -> None:
    """A `NaN` point short-circuits before the branches, so a null `p` does not null it.

    Probed through the private hook, since the public wrapper answers `NaN` on its own.
    """
    result = _null_p().select(r=Geometric(p=pl.col("p"))._pmf(pl.lit(math.nan)))["r"].item()
    assert math.isnan(result)
