"""A column-name `str` passed to a value-keyed method must mean `pl.col(name)`, not `pl.lit(name)`.

`as_expr` previously wrapped every non-`Expr` input in `pl.lit`, so `dist.pdf("x")` built the string
literal `"x"`; the numeric plugins then cast it to all-null, silently producing wrong answers. This
module pins the fix across every distribution and every value-keyed method: passing a column name as
a string is equivalent to passing `pl.col(name)`, and the result is not the all-null regression.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import Bernoulli, LogNormal, Normal, Uniform
from polars_stats.distributions._base import ContinuousDistribution

if TYPE_CHECKING:
    from polars_stats.distributions._base import _UnivariateDistribution

# Per-row column-parameterised instances; each reads its parameters from `FRAME`.
DISTRIBUTIONS: list[tuple[_UnivariateDistribution, str]] = [
    (Normal(mean="mu", std_dev="sigma"), "Normal"),
    (LogNormal(mu="mu", sigma="sigma"), "LogNormal"),
    (Uniform(min="lo", max="hi"), "Uniform"),
    (Bernoulli(p="p"), "Bernoulli"),
]

FRAME = pl.DataFrame(
    {
        "mu": [0.0, 1.0, -0.5],
        "sigma": [1.0, 2.0, 0.5],
        "lo": [0.0, -1.0, 2.0],
        "hi": [1.0, 3.0, 5.0],
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
    # Guard the actual regression: the string-arg path used to collapse to all-null.
    assert via_str.null_count() < via_str.len()
