from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.benchmark._bench import ROWS, length_frame, pointwise_frame
from tests.property._specs import ALL_SPECS

if TYPE_CHECKING:
    import polars as pl


@pytest.fixture(scope="session")
def bench_frame() -> pl.DataFrame:
    """A single `ROWS`-row frame for the sampler and summary benchmarks, which need only a length."""
    return length_frame(ROWS)


@pytest.fixture(scope="session")
def pointwise_frames() -> dict[str, pl.DataFrame]:
    """Per-distribution `value` / `quantile` input frames for the pointwise benchmarks, built once."""
    return {spec.name: pointwise_frame(spec, ROWS) for spec in ALL_SPECS}
