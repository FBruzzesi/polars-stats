# Test layout convention

Tests for the distribution surface live in four categories, split by what they assert.
A new distribution adds tests by following the existing files, not by writing fixtures from scratch.

## 1. Per-method behavioural tests — `tests/distributions/<name>/`

One directory per distribution, mirroring [`distributions/bernoulli/`](distributions/bernoulli),
one file per public method (`cdf_test.py`, `ppf_test.py`, `sample_test.py`, ...).
These assert *behaviour that is specific to the distribution* and is not captured by a numeric match against SciPy:

* support handling (clamping outside `[min, max]`, the `{0, 1}` mass points, infinite `ppf` endpoints);
* the three parameter modes — scalar, `pl.Expr`, mixed/`str` column — produce the right result;
* null propagation: a null in any input position yields a null output;
* invalid parameters raise `ComputeError` rather than silently computing a wrong value;
* sampler contracts: reproducibility under a fixed seed, per-row independence, `over` / `group_by`.

Shared inputs (a `seed`, a `frame` factory, a `value_grid`, parameter grids) live in the directory's `conftest.py` so
no test file redeclares them. Keep these files focused on *edges*, not on numeric parity — that is covered once, below.

## 2. SciPy parity — `tests/scipy_parity/<name>_test.py`

One table-driven module per distribution. Each lists its methods as `Case` rows and runs them through the shared harness
in [`_harness.py`](scipy_parity/_harness.py), which compares every method against the matching `scipy.stats` attribute
across a parameter grid.

Adding a method is one row; the comparison mechanics (grid selection, Boolean→float casting, tolerance)
are not re-implemented.

Tolerances follow the *upstream implementation*, not aspiration: closed-form methods hold the default `1e-12`;
methods routed through `erf`/`erfc` or a binary-search `inverse_cdf` pass a relaxed per-`Case` tolerance
(e.g. the normal cdf/sf family at `1e-9`). Document any exception inline on the `Case`.

## 3. Property tests — `tests/property/`

Distribution-agnostic invariants, written once and parametrised across every distribution via the `DistSpec` registry in
[`_specs.py`](property/_specs.py). A new distribution adds one `DistSpec` entry and inherits the whole suite:

* `0 <= cdf(x) <= 1` and `cdf` non-decreasing in `x`;
* `cdf(ppf(q)) ~= q` for `q` in the open unit interval (continuous);
* `pdf(x) >= 0` (continuous) / `pmf(x) >= 0` (discrete);
* trapezoidal integral of `pdf` over the (truncated) support `~= 1` (continuous);
* sum of `pmf` over the finite support `~= 1` (discrete);
* `sample(seed=N)` is reproducible across two calls.

These use [`hypothesis`](https://hypothesis.readthedocs.io).

The capped profile in [`conftest.py`](property/conftest.py) (`max_examples`, `deadline=None`) keeps the suite under the
CI budget and non-flaky; raise `max_examples` locally when investigating a failure.

## 4. Benchmark guard — `tests/benchmark/`

Times **our crate only** (no scipy or other contender: that comparison is the separate manual report under
[`benchmarks/`](../benchmarks), `make bench-compare`). One module per method group, all parametrised off the same
[`_specs.py`](property/_specs.py) `DistSpec` registry, so a distribution with a property-test row is benchmarked for
free:

* [`sample_bench_test.py`](benchmark/sample_bench_test.py): `sample` (scalar fast path + column per-row path) and the
    `samples` array path, the canonical per-row-RNG regression;
* [`pointwise_bench_test.py`](benchmark/pointwise_bench_test.py): pmf/pdf, log-density, cdf, log_cdf, sf, log_sf, ppf,
    isf, in both parameter regimes;
* [`summary_bench_test.py`](benchmark/summary_bench_test.py): mean, variance, std, median, entropy, column regime only.

Bodies are written against the `pytest-codspeed` `benchmark` fixture and carry the `benchmark` mark:
`make bench-guard` runs them locally (walltime mode, 1M rows, printed timing table), and the CI `bench-guard` job
runs the **same command** under the CodSpeed runner (deterministic instruction count). Row count is
`POLARS_STATS_BENCH_ROWS`. The default `make test` run deselects the mark (`-m "not benchmark"` in `addopts`) but still
imports the modules, so API drift is caught. The only registry input the property suite does not already supply is
`bench_params`, a fixed representative parameter tuple per `DistSpec`
(the suite samples a strategy; the guard needs one deterministic point).
