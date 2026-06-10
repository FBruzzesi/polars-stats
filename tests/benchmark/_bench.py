"""Shared inputs for the regression-guard benchmarks (`tests/benchmark/`).

Everything the three benchmark modules need that is not already on the property `DistSpec`: the row count,
the fixed seed / sample width, the input-frame builders, and the per-spec method enumeration. The guard
times our crate only, so there is nothing here about scipy or any other contender.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from polars_stats.distributions._base import _UnivariateDistribution
    from tests.property._specs import DistSpec

# Rows per benchmark body. The default is tuned for the (deterministic, instruction-count) CodSpeed CI
# run: a per-row-cost regression such as re-seeding the RNG every row shows at any N, so a small frame
# keeps the CI budget low without losing the signal. `make bench-guard` sets `POLARS_STATS_BENCH_ROWS=1_000_000`
# for stable local wall-clock numbers; override the env var for anything in between.
DEFAULT_BENCH_ROWS = 10_000
ROWS = int(os.environ.get("POLARS_STATS_BENCH_ROWS", DEFAULT_BENCH_ROWS))

# `samples` draws this many independent variates per row; the per-row sub-seed derivation is the regression
# this array path guards, so the width is fixed and modest.
SAMPLES_SIZE = 10

# One fixed seed across the suite: timings must not depend on it, and a constant keeps runs comparable.
SEED = 0

# Parameter regimes for pointwise methods and samplers: `scalar` exercises the constant-parameter fast path,
# `column` the general per-row plugin (where per-row scale matters).
REGIMES = ("scalar", "column")

# Summary statistics, benchmarked in the column regime only: with scalar params they collapse to a length-1
# scalar, so per-row parameters are the only meaningful input (see issue-tracker/03-infra-benchmarks.md).
SUMMARY_METHODS = ("mean", "variance", "std", "median", "entropy")


def length_frame(n: int) -> pl.DataFrame:
    """An `n`-row frame whose only job is to set `pl.len()` for the samplers and column-param expansion."""
    return pl.select(pl.int_range(0, n, dtype=pl.Int64).alias("row"))


def pointwise_frame(spec: DistSpec, n: int) -> pl.DataFrame:
    """`n` rows with a `value` column over the spec's eval window and a `quantile` column in (0, 1).

    `value` feeds the density / cdf / sf family; `quantile` feeds ppf / isf. Built once per spec so a
    benchmark times only the method call, not the input construction.
    """
    lo, hi = spec.eval_range(spec.bench_params)
    frac = pl.int_range(0, n, dtype=pl.Int64).cast(pl.Float64) / float(max(n - 1, 1))
    return pl.select(value=lo + frac * (hi - lo), quantile=0.01 + frac * 0.98)


def make_dist(spec: DistSpec, regime: str) -> _UnivariateDistribution:
    """The spec's instance in the requested regime: `scalar` (fast path) or `column` (per-row)."""
    return spec.make(spec.bench_params) if regime == "scalar" else spec.make_columns(spec.bench_params)


def pointwise_methods(spec: DistSpec) -> list[tuple[str, str]]:
    """`(method_name, input_column)` for every pointwise method the spec exposes.

    `input_column` names the `pointwise_frame` column each method reads: `value` for the density / cdf / sf
    family, `quantile` for ppf / isf.
    """
    density = "pdf" if spec.continuous else "pmf"
    return [
        (density, "value"),
        (f"log_{density}", "value"),
        ("cdf", "value"),
        ("log_cdf", "value"),
        ("sf", "value"),
        ("log_sf", "value"),
        ("ppf", "quantile"),
        ("isf", "quantile"),
    ]
