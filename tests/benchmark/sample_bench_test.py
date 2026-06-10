"""Regression-guard benchmarks for the sampler hot paths (`sample`, `samples`), our crate only.

Written against the pytest-codspeed ``benchmark`` fixture: deterministic instruction count under the
CodSpeed runner in CI, walltime locally. Both regimes are timed: ``scalar`` parameters take the
constant-parameter fast path, ``column`` parameters the general per-row plugin. The per-row RNG
construction cost scales with the row count, so a single wide column surfaces a regression like the
ChaCha20-per-row one this guard was written for.

Run with ``make bench-guard``; the default run deselects the ``benchmark`` mark.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.benchmark._bench import REGIMES, SAMPLES_SIZE, SEED, make_dist
from tests.property._specs import ALL_SPECS

if TYPE_CHECKING:
    from collections.abc import Callable

    import polars as pl

    from tests.property._specs import DistSpec

pytestmark = pytest.mark.benchmark


@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.name)
def test_sample(benchmark: Callable[..., object], bench_frame: pl.DataFrame, spec: DistSpec, regime: str) -> None:
    """`sample`: one variate per row, in the scalar (fast-path) and column (per-row) regimes."""
    expr = make_dist(spec, regime).sample(seed=SEED)
    benchmark(lambda: bench_frame.select(s=expr))


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.name)
def test_samples_array(benchmark: Callable[..., object], bench_frame: pl.DataFrame, spec: DistSpec) -> None:
    """`samples`: `SAMPLES_SIZE` independent draws per row, the canonical per-row-RNG regression guard."""
    expr = make_dist(spec, "scalar").samples(SAMPLES_SIZE, seed=SEED)
    benchmark(lambda: bench_frame.select(s=expr))
