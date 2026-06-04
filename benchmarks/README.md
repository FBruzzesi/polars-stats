# Benchmarks: `polars_stats` vs `scipy.stats`

Manual comparison of the **sampling** methods against `scipy.stats`, on speed and peak memory.

This is the routine whose output backs the comparison numbers in the project README/documentation benchmarks.

It is **not** the CI regression guard.

Scope is deliberately narrow: `sample` (one variate per row) and `samples` (`n_samples` per row) against
`scipy.rvs`.

## Running

There is one runnable entrypoint, [run.py](run.py). Run it with `uv` and the `benchmarks` dependency
group, which provides the extra tooling (`cyclopts`, `rich`, `psutil`, plus `scipy` / `numpy`):

```bash
uv run --group benchmarks benchmarks/run.py                                   # all distributions, rich table in the terminal
uv run --group benchmarks benchmarks/run.py normal binomial                   # a subset
uv run --group benchmarks benchmarks/run.py normal --rows 1_000_000 10_000_000 --n-samples 5 10 20  # sweep a grid
uv run --group benchmarks benchmarks/run.py --format markdown                 # write benchmarks/results/<dist>.md
uv run --group benchmarks benchmarks/run.py --format json                     # write benchmarks/results/<dist>.json
uv run --group benchmarks benchmarks/run.py --help
```

| argument | default | meaning |
| --- | --- | --- |
| `distributions` (positional) | all | which distributions to compare, e.g. `normal binomial` |
| `--rows` | `1_000_000` | one or more row counts to sweep; rows for `sample`/`samples`/`scipy.rvs` |
| `--n-samples` | `10` | one or more draws-per-row widths to sweep for `samples` |
| `--iterations` | `50` | timed runs per cell; runtime reported as `p50 ± std` |
| `--seed` | `0` | seed for the samplers (reproducible) |
| `--format` | `rich` | `rich` (coloured terminal table), `markdown`, or `json` |
| `--output-dir` | `benchmarks/results/` | where `markdown` / `json` files are written |

`--rows` and `--n-samples` accept multiple values and are swept as a grid in one report: `sample` is
benchmarked once per `rows` value (`n_samples` does not apply, shown as `-`); `samples` over the full
`rows` x `n_samples` product.

Output formats:

* `rich`: a coloured table on the terminal (speedup green when `polars_stats` wins). Nothing is written.
* `markdown`: a `## <dist>` document with an environment stamp and table. Written to
  `benchmarks/results/<dist>.md`.
* `json`: machine-readable (environment, config, per-method timing and memory). Written to
  `benchmarks/results/<dist>.json`.

`benchmarks/results/` is git-ignored: the files are machine-specific. Commit curated tables into the
README or a docs page, not the raw per-run output.

## Build mode (important for fair numbers)

`uv run --group benchmarks` measures whatever `polars_stats` is installed in the project environment.

Build it in **release** mode first (`make install-release`, i.e. `maturin develop --release`). A debug
`maturin develop` build runs the Rust extension unoptimised and would make `polars_stats` look far
slower than scipy's optimised C, invalidating the comparison.

## Layout

* [run.py](run.py): the single runnable script and `cyclopts` CLI. Holds the `REGISTRY` of distributions
  (each a `Comparison`: the `polars_stats` instance and the matching frozen `scipy.stats` distribution,
  reparameterised to scipy's convention) and dispatches to the harness.
* [_harness.py](_harness.py): the comparison routine. Builds the calls, times them, measures peak memory
  in isolated subprocesses, checks shapes, renders the report. Imported, never run directly.

Adding a distribution is one entry in `REGISTRY`.

## What is measured

* **Time:** wall-clock of the full native call, as **median ± standard deviation** (ms) over
  `--iterations` runs, after one warmup. Measured in-process; the reported speedup is
  `scipy p50 / polars_stats p50`. The `polars_stats` side runs on the lazy **streaming** engine
  (`collect(engine="streaming")`), the chunked path users hit at scale.
* **Peak memory:** peak resident-set growth (MiB) of a single call, each measured **in a fresh spawned subprocess**.
  Isolation is required: an in-process measurement after the timing loop is meaningless because each library's
  allocator retains freed pages differently (scipy would read ~0 while `polars_stats` re-allocates).
  RSS (not `tracemalloc`) so the native Rust/Arrow and NumPy allocations are counted.
* **Correctness gate:** values cannot match across independent RNGs, so the gate is a shape check
  (column length and array width vs scipy's output shape), flagged `MISMATCH` if it diverges. Warn-only.

Caveats on the memory numbers (read before quoting them):

* They are **approximate, same-machine relative** figures, not exact footprints. Peak RSS is coarse and
  includes the interpreter and imported libraries' working set.
* The `polars_stats` figures include **one-time query-engine init** (thread pool, etc.) that the first
  polars operation in a process pays once; a long-running process amortises it away. This inflates the
  `polars_stats` memory at small `--rows`, where the fixed init dominates the output size.
* One measurement per call (not a distribution); allocation sizes are deterministic, but the sampled
  peak can miss a very short transient.

Out of scope (this routine is sampling-only): pointwise methods (`pdf`/`cdf`/`ppf`/...), summary
statistics, and the column-parameter regime.
