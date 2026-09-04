# Benchmarks: `polars_stats` vs `scipy.stats`

Manual comparison against `scipy.stats`, on speed and peak memory.

This is **not** a CI regression guard, and nothing it produces is published today: the project README and `docs/`
deliberately carry no performance claims. The goal of the comparison is to *locally* test for regressions or
performance improvements tweaking the internals of an existing implementation.

In the future, it might be used to report a comparison.

Three method families are compared, one row per method in the `METHOD_SPECS` table in [_harness.py](_harness.py):

* **sampling**: `sample` (one variate per row) and `samples` (`n_samples` per row) against `scipy.rvs`;
* **value-keyed**: `density` (`pdf` / `pmf` by family), `log_density`, `cdf`, `log_cdf`, `sf`, `log_sf`, `ppf` and `isf`
  against the matching frozen-scipy methods, both sides evaluating the same deterministic inputs
  (the distribution's own seeded draws; uniform quantiles for `ppf` / `isf`);
* **moments**: `mean`, `variance`, `std` and `entropy` against the frozen-scipy moments. These take no value column,
  so whether they scale with `--rows` depends on the regime.

## Parameter regimes

Every cell is measured in one of three **regimes**, which is how the parameters are supplied.
They are three different code paths on both sides, so a report labels every row with its regime:

| regime      | `polars_stats` | `scipy`             | what it exercises |
| ----------- | -------------- | ------------------- | ----------------- |
| `scalar`    | Python numbers | plain scalars       | constant-parameter **fast path**: parameters validated once and passed as plugin kwargs |
| `column`    | `pl.col(...)`  | length-`rows` array | the **general per-row path**, one parameter value per row |
| `broadcast` | `pl.lit(...)`  | length-1 array      | the per-row path fed a length-1 input, which `align_inputs` broadcasts up to the call's row count |

**Never compare a cell against one in another regime**: a `column` cell does strictly more work per row by design,
so the comparison is meaningless and flattering in whichever direction you read it.
The regime is part of every case's id and a column of every report so the two cannot be diffed by accident.
Within a regime the comparison is fair, and `column` is the honest one to quote for a per-row workload.

The regime also decides whether a **moment** is a throughput comparison: with `scalar` or `broadcast` parameters both
sides return a single value and are O(1), so `--rows` does not scale the work: those cells track per-call overhead only,
and the reported ratio is mostly `collect()` dispatch rather than moment arithmetic.
With `column` parameters both sides return one value per row and are O(n).

Before quoting a `column`-regime moment ratio, know what the scipy side is paying. The frozen API's
own broadcast/validate/mask machinery dominates those cells, and `entropy` is worse: `rv_generic.entropy`
always dispatches through `np.vectorize(self._entropy)`, *even when `_entropy` already handles arrays*.
`beta._entropy` is a vectorised `_lazyselect` over four asymptotic branches, so scipy runs that whole
machinery once per element, around what has been a closed form since scipy 1.11. That is why
`beta.entropy` at 1M rows takes so long, and why that one cell is the single largest contributor to a
full sweep's runtime.

See "Keeping the scipy side native" below before reading anything into those numbers.

## Running

There is one runnable entrypoint, [run.py](run.py). Run it with `uv` and the `benchmarks` dependency
group, which provides the extra tooling (`cyclopts`, `rich`, `psutil`, plus `scipy` / `numpy`):

```bash
uv run --group benchmarks benchmarks/run.py  # all distributions, rich table in the terminal
uv run --group benchmarks benchmarks/run.py normal binomial  # a subset of distributions
uv run --group benchmarks benchmarks/run.py --methods sample density ppf  # a subset of methods
uv run --group benchmarks benchmarks/run.py --regimes scalar column  # a subset of regimes
uv run --group benchmarks benchmarks/run.py normal --rows 1_000_000 10_000_000 --n-samples 5 10 20  # sweep a grid
uv run --group benchmarks benchmarks/run.py --memory  # also measure peak RSS (slow: a subprocess per cell per side)
uv run --group benchmarks benchmarks/run.py --format markdown  # write benchmarks/results/<dist>.md
uv run --group benchmarks benchmarks/run.py --format json  # write benchmarks/results/<dist>.json
uv run --group benchmarks benchmarks/run.py --help
```

| argument | default | meaning |
| --- | --- | --- |
| `distributions` (positional) | all | which distributions to compare, e.g. `normal binomial` |
| `--methods` | all | which methods to compare: `sample`, `samples`, `density` (= `pdf`/`pmf`), `log_density`, `cdf`, `log_cdf`, `sf`, `log_sf`, `ppf`, `isf`, `mean`, `variance`, `std`, `entropy` |
| `--regimes` | all | which parameter regimes to compare: `scalar`, `column`, `broadcast` |
| `--rows` | `1_000_000` | one or more row counts to sweep; rows for every method |
| `--n-samples` | `10` | one or more draws-per-row widths to sweep for `samples` |
| `--max-iterations` | `50` | **upper bound** on timed runs per cell |
| `--max-seconds` | `30.0` | wall-clock budget for one cell's timed loop |
| `--min-iterations` | `3` | lower bound on timed runs per cell, applied even when the budget is up |
| `--memory` | off | also measure peak RSS per cell per side, in an isolated subprocess |
| `--seed` | `0` | seed for the samplers, the evaluation inputs and the parameter draws (reproducible) |
| `--format` | `rich` | `rich` (coloured terminal table), `markdown`, or `json` |
| `--output-dir` | `benchmarks/results/` | where `markdown` / `json` files are written |

(see `uv run --group benchmarks benchmarks/run.py --help`).

Everything above `--memory` is a field of `Sweep` (and its nested `Budget`) in [_harness.py](_harness.py),
which the CLI flattens: the dataclass owns the defaults, the help text and the validation, so there is one declaration
rather than a signature that restates them.

`--methods`, `--regimes`, `--rows` and `--n-samples` accept multiple values and are swept as a grid in one report: every
method except `samples` is benchmarked once per regime per `rows` value (`n_samples` does not apply, reported as `-`);
`samples` over the full `rows` x `n_samples` product.

Regime is the outermost axis, so each regime's rows are contiguous. A repeated value on any of those flags is rejected
rather than silently measuring the same cell twice. A progress line names the case in flight, so a long sweep is visibly
running rather than hung.

**A cell is measured under a time budget, not a fixed iteration count**: cell costs span orders of magnitude:
a `scalar` moment costs microseconds while a `column`-regime `entropy` for a distribution whose entropy scipy does not
vectorise runs into **tens of seconds per call** at 1M rows, well past the default `--max-seconds`.
A count high enough for the cheap cells makes the expensive ones unrunnable, which is how a sweep gets abandoned
half-run. So the warmup call is timed and used to plan the iteration count, `--max-seconds` then caps the loop,
and `--min-iterations` guarantees a meaningful median either way.

A cell whose *single* call already costs more than the whole budget is measured exactly once,
since `--min-iterations` of those would blow the budget several times over.

**Read the `iters` column before quoting a cell**: it carries the count each side actually ran, and a cell that fell
back to one or to `--min-iterations` says so there. At one iteration the spread is reported as `-` (`null` in JSON)
rather than a flattering `0.000`; above one it is a sample standard deviation (`ddof=1`).

A cell that raises is reported as `SKIPPED` and the sweep continues, and an interrupt still writes the cells already
measured. A long sweep therefore cannot lose an hour of completed work to one bad cell.

Output formats:

* `rich`: a coloured table on the terminal (ratio green when `polars_stats` wins). Nothing is written.
* `markdown`: a `## <dist>` document with an environment stamp and table. Written to `benchmarks/results/<dist>.md`.
* `json`: machine-readable (schema version, environment, sweep config, per-cell timing and memory).
  Written to `benchmarks/results/<dist>.json`.

Both file formats stamp the rows, methods, regimes and budget the run actually used, so a narrower re-run cannot be
mistaken for a wider one. `benchmarks/results/` is git-ignored: the files are machine-specific, and a run overwrites
the previous file for the same distribution.

If numbers are ever published, commit a curated table rather than the raw per-run output.

## Build mode (enforced)

`uv run --group benchmarks` measures whatever `polars_stats` is installed in the project environment.

Build it in **release** mode first (`make install-release`, i.e. `maturin develop --release`).
A debug `maturin develop` build runs the Rust extension unoptimised and would make `polars_stats` look far slower than
scipy's optimised C, invalidating the comparison.

This is checked, not just documented, because a debug run produces a table that looks entirely normal.
`src/lib.rs` exposes `cfg!(debug_assertions)` as `polars_stats._internal.__debug_build__`, and
`require_release_build()` refuses to start otherwise - including against an extension too old to carry the flag,
since it cannot vouch for itself. The profile is also stamped into every report header and the JSON `environment`,
so a saved report always says which build produced it.

## Layout

* [run.py](run.py): the single runnable script and `cyclopts` CLI. Holds the `REGISTRY` of distributions and drives the
  harness. Each entry is a `Comparison`: a `ParamSpec` per parameter (the fixed value the `scalar` and `broadcast`
  regimes use, plus the domain the `column` regime draws from) and a **factory** turning one regime's realised
  parameters into both sides' distributions, applying scipy's reparameterisation. A factory rather than a frozen
  instance pair is what lets one registry serve all three regimes; there is deliberately no second registry and
  no second entry point.
* [_harness.py](_harness.py): the comparison routine. Builds the calls, times them, measures peak memory in isolated
  subprocesses, gates correctness, renders the report. Imported, never run directly. Its seams are `METHOD_SPECS` (the
  method metadata table), `Case` (one measurable cell, as frozen and picklable data carrying its own `id`),
  `Sweep` (a grid, and the CLI's option surface), and `Contender` (a `(Comparison, Case) -> Call` builder registered by
  name in `CONTENDERS`, with ratios taken against `REFERENCE_CONTENDER`).

Adding a distribution is one entry in `REGISTRY`: its `ParamSpec`s and its factory.
Adding a method is a token in `Method` plus a row in `METHOD_SPECS`, and the import-time check that pairs the two will
fail the run if you add only one.

The correctness gate checks itself at import, for the same reason. `_outputs_agree` is the harness's only correctness
signal and its failure mode is silent - a broken gate reports `ok` for every cell forever - so `_check_gate_can_reject()`
runs six cases on synthesised frames (a real match, a value past the tolerance, a short reference, a null reference,
matching draws, a wrong draw width) and refuses to import if any verdict is wrong.
The passing case is part of it deliberately: a gate stuck on `False` would satisfy the rejections.
It costs well under a millisecond and needs neither the plugin nor scipy.

A `ParamSpec` domain must be chosen so that **every** draw is a valid parameterisation. A joint constraint like
`min < max` is expressed as two non-overlapping domains rather than a rejection step, so realisation never has to retry.
Parameter draws are derived from `(regime, rows, seed)` alone, so both sides and the memory subprocesses regenerate
identical arrays; they take their own seed stream, so widening a domain cannot shift the evaluation inputs.

An integer parameter must carry `integer=True`; `Params.plugin_int` checks the realised dtype, so a missing flag raises
in every regime rather than surfacing deep inside the plugin.

A `Comparison.build` must be a module-level function, checked on construction, because a lambda or closure would not
survive pickling into the memory subprocess.

## What is measured

* **Time:** wall-clock of the full native call, as **median +/- sample standard deviation** (ms) over as many iterations
  as the budget allowed, after one warmup. Measured in-process; the reported ratio is `scipy p50 / polars_stats p50`.
  The `polars_stats` side runs on the lazy **streaming** engine (`collect(engine="streaming")`), the chunked path users
  hit at scale, and its timed closure includes both query-plan construction and `collect()` dispatch where scipy's bound
  method has no equivalent.
  That asymmetry is a fixed per-call cost, so it is negligible on a large cell and dominant on the cheapest ones.
* **Peak memory:** **opt-in, via `--memory`**, because it spawns one subprocess per contender per cell and that cost
  dominates a full sweep. Peak resident-set growth (MiB) of a single call, each measured **in a fresh spawned subprocess**.
  Isolation is required: an in-process measurement after the timing loop is meaningless because each library's allocator
  retains freed pages differently (scipy would read ~0 while `polars_stats` re-allocates).
  RSS (not `tracemalloc`) so the native Rust/Arrow and NumPy allocations are counted.
  Without `--memory` the MiB columns are omitted entirely.
* **Correctness gate:** for the samplers, values cannot match across independent RNGs, so the gate is a shape check
  (column length and array width vs scipy's output shape). The value-keyed methods evaluate the same inputs on both
  sides, so their gate is `np.allclose` to loose tolerances (the scipy-parity test suite owns the tight per-method bounds).
  Moments are gated on both sides' shapes: height 1 for `scalar` and `broadcast`, `rows` for `column`.
  A null on the `polars_stats` side fails the gate outright, because `to_numpy` renders it as `NaN` and the comparison
  treats `NaN` as agreeing with scipy's own. Flagged `MISMATCH` if it diverges. Warn-only.

Caveats on the memory numbers (read before quoting them):

* They are **approximate, same-machine relative** figures, not exact footprints. Peak RSS is coarse and
  includes the interpreter and imported libraries' working set.
* The `polars_stats` figures include **one-time query-engine init** (thread pool, etc.) that the first
  polars operation in a process pays once; a long-running process amortises it away. This inflates the
  `polars_stats` memory at small `--rows`, where the fixed init dominates the output size.
* One measurement per call (not a distribution); allocation sizes are deterministic, but the sampled
  peak can miss a very short transient. The sampler polls every 0.5 ms, which is coarser than the whole
  call on a `scalar` moment cell, so those readings are near-baseline rather than a real peak.

## Keeping the scipy side native

Both sides must be measured on the path a real user of that library would take, and two of those
choices are invisible in the output.

**RNG family (samplers)**: scipy's samplers are handed a `numpy.random.Generator`, not an int seed.
`random_state=<int>` makes scipy build a legacy `RandomState` and draw from **MT19937**, while our Rust side draws from
`Pcg64Mcg` ([src/rng.rs](../src/rng.rs)) - so an int seed compares a modern PCG generator against a 1997 Mersenne Twister.
That is an artefact of the seed's *type*, not of the API a scipy user writes, and it was worth 1.1x to 2.4x depending on
the distribution. Worse, it was not a uniform offset: it reordered the distributions, flattering `sample` on `normal`
(measured 6.2x before, 2.4x after) and `discrete_uniform` (2.6x before, 1.3x after) while *understating* `uniform`.
The generator is constructed per call, so scipy pays the same construct-from-seed cost our side does and
every timed iteration still draws identical values; construction is ~3 us.

**API generation (everything)**: the harness measures the **classic frozen API** (`norm(loc, scale)` then `.pdf(x)`),
which is what the overwhelming majority of scipy code written today looks like.
That is a deliberate, defensible baseline, but it is not scipy at its fastest.
Measured against the newer random-variable API (`scipy.stats.Normal`, `make_distribution`, scipy >= 1.15) on `normal`
with column parameters: the new API is roughly 1.3x to 1.9x faster on the value-keyed methods, effectively free for
`mean` and `variance` (it returns the parameter array rather than computing through the generic machinery),
and **~700x** faster on `beta.entropy`, where the frozen path falls back to a per-element `np.vectorize` loop.
So read every ratio as *against the classic frozen API*, and never quote one as "faster than scipy" without that
qualifier.

A `scipy-new` contender would be one registration in `CONTENDERS`, but it would not be a like-for-like baseline *today*:
the new API ships native classes for only a few families (`Normal`, `Uniform`, `Binomial`), so most of the registry
would have to go through `make_distribution`, and the location-scale families do not accept `loc` / `scale` there at all.
The frozen API therefore stays the baseline.
The numbers above exist to size what that choice costs, not to suggest switching.

One further scipy constraint worth knowing, since it is invisible in the output: with array parameters,
`rvs(size=(rows, draws))` cannot broadcast against the length-`rows` parameters, so the `column` regime draws
`size=(draws, rows)` and transposes back to the layout the `polars_stats` side returns. The transpose is a view rather
than a copy, which means scipy never pays for the row-major layout our side materialises.
The `column`-regime `samples` ratio is optimistic for scipy by roughly one pass over the output.
