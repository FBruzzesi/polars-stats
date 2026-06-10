"""Regression-guard benchmarks for the pointwise methods (pdf/pmf, log-density, cdf, sf, ppf, isf).

Our crate only, scalar + column parameter regimes, every method driven off the property `DistSpec`
registry so a new distribution is benchmarked the moment it has a spec row. Written against the
pytest-codspeed ``benchmark`` fixture (CodSpeed instruction count in CI, walltime locally); run with
``make bench-guard``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from tests.benchmark._bench import REGIMES, make_dist, pointwise_methods
from tests.property._specs import ALL_SPECS

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.property._specs import DistSpec

pytestmark = pytest.mark.benchmark

# One leaf benchmark per (distribution, regime, method): the registry's method enumeration keeps the
# discrete (pmf/log_pmf) vs continuous (pdf/log_pdf) split correct without per-distribution code here.
_Case = tuple["DistSpec", str, str, str]
_CASES: list[_Case] = [
    (spec, regime, method, column)
    for spec in ALL_SPECS
    for regime in REGIMES
    for method, column in pointwise_methods(spec)
]


def _case_id(case: _Case) -> str:
    spec, regime, method, _column = case
    return f"{spec.name}-{method}-{regime}"


@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_pointwise(
    benchmark: Callable[..., object],
    pointwise_frames: dict[str, pl.DataFrame],
    case: _Case,
) -> None:
    """Time one pointwise method on a `ROWS`-row input column in the given parameter regime."""
    spec, regime, method, column = case
    frame = pointwise_frames[spec.name]
    expr = getattr(make_dist(spec, regime), method)(pl.col(column))
    benchmark(lambda: frame.select(r=expr))
