"""Performance benchmarks for the Bernoulli sampler hot paths.

Written against the ``pytest-benchmark`` ``benchmark`` fixture so the runner can be
swapped for ``pytest-codspeed`` (instruction-count, deterministic) once CI allows it,
without touching the benchmark bodies.

These guard the per-row RNG construction cost: it scales with the number of rows, so a
single wide column is enough to surface a regression like the ChaCha20-per-row one.

Run with: ``pytest benchmarks --benchmark-only`` (see the ``benchmark`` Make target).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import polars_stats as ps

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

N_ROWS = 1_000_000
ARRAY_SIZE = 50
SEED = 0


@pytest.fixture(scope="module")
def frame() -> pl.DataFrame:
    # Materialised once: each benchmark times only the sampling, not frame construction.
    return pl.DataFrame(
        {
            "x": range(N_ROWS),
            "p": [0.3] * N_ROWS,
        }
    )


def test_sample_scalar_p(benchmark: Callable[..., object], frame: pl.DataFrame) -> None:
    # Scalar p expands to pl.repeat(p, n=pl.len()); the common path.
    expr = ps.Bernoulli(p=0.3).sample(seed=SEED)
    benchmark(lambda: frame.with_columns(b=expr))


def test_sample_column_p(benchmark: Callable[..., object], frame: pl.DataFrame) -> None:
    # Column p skips the repeat and feeds a real per-row probability series.
    expr = ps.Bernoulli(p=pl.col("p")).sample(seed=SEED)
    benchmark(lambda: frame.with_columns(b=expr))


def test_samples_array(benchmark: Callable[..., object], frame: pl.DataFrame) -> None:
    # Array path: ARRAY_SIZE independent draws per row.
    expr = ps.Bernoulli(p=0.3).samples(ARRAY_SIZE, seed=SEED)
    benchmark(lambda: frame.with_columns(b=expr))
