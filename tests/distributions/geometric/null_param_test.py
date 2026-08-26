"""A null `p` nulls every answer that depends on it, and only those.

`pmf(0) = 0`, `cdf(0) = 0` and `sf(0) = 1` are off-support constants with no `p` in them, so they
survive a null one; the class docstring promises it and `Bernoulli` behaves the same way.

It is a placement contract on a single gate. Every closed-form method reads `p` through one
validating round trip (`Geometric._when_p_valid`) rather than naming the validated `p` inline, and
widening that gate from the `p`-dependent branch to the whole method would null these constants and
still leave the suite green: the per-method null tests all evaluate on the support.
"""

from __future__ import annotations

import polars as pl
import pytest

from polars_stats import Geometric

_NEG_INF = float("-inf")

# (id, expr builder, expected) for the answers that hold with no `p` at all.
_OFF_SUPPORT: list[tuple[str, str, float, float]] = [
    ("pmf", "pmf", -1.0, 0.0),
    ("pmf", "pmf", 0.0, 0.0),
    ("pmf", "pmf", 0.5, 0.0),
    ("log_pmf", "log_pmf", 0.0, _NEG_INF),
    ("cdf", "cdf", 0.0, 0.0),
    ("log_cdf", "log_cdf", 0.0, _NEG_INF),
    ("sf", "sf", 0.0, 1.0),
    ("log_sf", "log_sf", 0.0, 0.0),
]

_ON_SUPPORT = ["pmf", "log_pmf", "cdf", "log_cdf", "sf", "log_sf"]

_PARAMETER_ONLY = ["mean", "variance", "std", "median", "entropy"]


@pytest.mark.parametrize(
    ("method", "value", "expected"),
    [(m, v, e) for _, m, v, e in _OFF_SUPPORT],
    ids=[f"{name}({value})" for name, _, value, _ in _OFF_SUPPORT],
)
def test_off_support_constant_survives_a_null_p(method: str, value: float, expected: float) -> None:
    frame = pl.DataFrame({"p": [None]}, schema={"p": pl.Float64})
    result = frame.select(r=getattr(Geometric(p=pl.col("p")), method)(value))["r"].item()
    assert result == expected


@pytest.mark.parametrize("method", _ON_SUPPORT)
def test_on_support_value_nulls_under_a_null_p(method: str) -> None:
    frame = pl.DataFrame({"p": [None]}, schema={"p": pl.Float64})
    assert frame.select(r=getattr(Geometric(p=pl.col("p")), method)(1.0))["r"].item() is None


@pytest.mark.parametrize("method", _PARAMETER_ONLY)
def test_moment_nulls_under_a_null_p(method: str) -> None:
    frame = pl.DataFrame({"p": [None]}, schema={"p": pl.Float64})
    assert frame.select(r=getattr(Geometric(p=pl.col("p")), method)())["r"].item() is None


@pytest.mark.parametrize("method", ["ppf", "isf"])
@pytest.mark.parametrize("quantile", [0.5, 2.0])
def test_inverse_nulls_under_a_null_p(method: str, quantile: float) -> None:
    """Null inside `[0, 1]` because the answer needs `p`, and null outside it by the domain contract."""
    frame = pl.DataFrame({"p": [None]}, schema={"p": pl.Float64})
    assert frame.select(r=getattr(Geometric(p=pl.col("p")), method)(quantile))["r"].item() is None
