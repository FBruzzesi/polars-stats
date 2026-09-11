"""A column-name `str` passed to a value-keyed method means `pl.col(name)`, never `pl.lit(name)`.

A string literal would reach the numeric plugins and be cast to all-null, so every case also asserts the
result is not all-null.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import (
    Bernoulli,
    Beta,
    Binomial,
    DiscreteUniform,
    Exponential,
    Geometric,
    LogNormal,
    Normal,
    Uniform,
)
from polars_stats.distributions._base import ContinuousDistribution

if TYPE_CHECKING:
    from polars_stats.distributions._base import _UnivariateDistribution

# Per-row column-parameterised instances; each reads its parameters from `FRAME`.
DISTRIBUTIONS: list[tuple[_UnivariateDistribution, str]] = [
    (Normal(mu="mu", sigma="sigma"), "Normal"),
    (LogNormal(mu="mu", sigma="sigma"), "LogNormal"),
    (Uniform(min="lo", max="hi"), "Uniform"),
    (Bernoulli(p="p"), "Bernoulli"),
    (Binomial(n="n", p="p"), "Binomial"),
    (DiscreteUniform(min="n", max="n"), "DiscreteUniform"),  # min == max is a valid point mass
    (Geometric(p="p"), "Geometric"),
    (Exponential(rate="p"), "Exponential"),  # `p` is positive on every row, a valid rate
    (Beta(a="p", b="sigma"), "Beta"),  # both columns are positive on every row, valid shapes
]

FRAME = pl.DataFrame(
    {
        "mu": [0.0, 1.0, -0.5],
        "sigma": [1.0, 2.0, 0.5],
        "lo": [0.0, -1.0, 2.0],
        "hi": [1.0, 3.0, 5.0],
        "n": [5, 10, 20],
        "p": [0.2, 0.5, 0.9],
        "x": [0.5, 1.5, 3.0],  # support points for pdf / pmf / cdf / sf
        "q": [0.1, 0.5, 0.9],  # quantiles in [0, 1] for ppf / isf
    }
)

# (method name, column the method reads). Value-keyed methods read `x`; inverse methods read `q`.
SHARED_METHODS = [("cdf", "x"), ("log_cdf", "x"), ("sf", "x"), ("log_sf", "x"), ("ppf", "q"), ("isf", "q")]
CONTINUOUS_METHODS = [("pdf", "x"), ("log_pdf", "x")]
DISCRETE_METHODS = [("pmf", "x"), ("log_pmf", "x")]


def _cases() -> tuple[list[tuple[_UnivariateDistribution, str, str]], list[str]]:
    cases, ids = [], []
    for dist, dist_id in DISTRIBUTIONS:
        extra = CONTINUOUS_METHODS if isinstance(dist, ContinuousDistribution) else DISCRETE_METHODS
        for method, column in SHARED_METHODS + extra:
            cases.append((dist, method, column))
            ids.append(f"{dist_id}.{method}")
    return cases, ids


_CASES, _IDS = _cases()


@pytest.mark.parametrize(("dist", "method", "column"), _CASES, ids=_IDS)
def test_str_value_arg_equals_col_expr(dist: _UnivariateDistribution, method: str, column: str) -> None:
    via_str = FRAME.select(r=getattr(dist, method)(column))["r"]
    via_expr = FRAME.select(r=getattr(dist, method)(pl.col(column)))["r"]
    assert_series_equal(via_str, via_expr)
    assert via_str.null_count() < via_str.len()
