"""Regression-guard benchmarks for the summary statistics (mean, variance, std, median, entropy).

Column-param regime only: with scalar params a summary collapses to a length-1 scalar, so per-row
parameters (a different distribution per row) are the only input that scales. Our crate only, driven off
the property `DistSpec` registry. Written against the pytest-codspeed ``benchmark`` fixture; run with
``make bench-guard``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.benchmark._bench import SUMMARY_METHODS, make_dist
from tests.property._specs import ALL_SPECS

if TYPE_CHECKING:
    from collections.abc import Callable

    import polars as pl

    from tests.property._specs import DistSpec

pytestmark = pytest.mark.benchmark

_Case = tuple["DistSpec", str]
_CASES: list[_Case] = [(spec, method) for spec in ALL_SPECS for method in SUMMARY_METHODS]


def _case_id(case: _Case) -> str:
    spec, method = case
    return f"{spec.name}-{method}"


@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_summary(benchmark: Callable[..., object], bench_frame: pl.DataFrame, case: _Case) -> None:
    """Time one summary statistic with per-row (column) parameters over a `ROWS`-row frame."""
    spec, method = case
    expr = getattr(make_dist(spec, "column"), method)()
    benchmark(lambda: bench_frame.select(r=expr))
