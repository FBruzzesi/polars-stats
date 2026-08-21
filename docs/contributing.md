---
icon: lucide/hammer
---

# Contributing

## Build from source

The package is a compiled Rust extension built with `maturin`, with `uv` managing the environment and lockfile.

```bash
make install           # uvx maturin develop
make install-release   # uvx maturin develop --release (use for benchmarking)
```

## Day-to-day commands

Prefer the `Makefile` targets so flags stay consistent with CI:

```bash
make test       # POLARS_MAX_THREADS=4 uv run --group testing pytest tests
make typing     # pyrefly + pyright + mypy (all three, as in CI)
make lint       # prek hooks (ruff, rumdl, ryl) + cargo fmt (nightly) + clippy
make benchmark  # polars_stats vs scipy comparison report (benchmarks/)
```

`make test` caps `POLARS_MAX_THREADS=4` on purpose: it forces multi-thread, multi-chunk execution so the chunk- and
thread-invariance of `sample` actually gets exercised.

It does not pin a query engine, so it runs whichever one your environment resolves (`POLARS_ENGINE_AFFINITY`, else the
polars default). CI runs both, because the two chunk a plugin's inputs differently: in-memory calls it once over the
whole column, streaming once per morsel. A chunk-boundary or input-length bug can therefore pass under one and fail
under the other. Pin an engine with:

```bash
POLARS_ENGINE_AFFINITY=in-memory uv run --group testing pytest tests
POLARS_ENGINE_AFFINITY=streaming uv run --group testing pytest tests
```

To preview these docs locally:

```bash
uv run --group docs zensical serve
```

## Repository layout

```text
polars-stats/
├── Cargo.toml
├── pyproject.toml
├── rust-toolchain.toml
├── src/
│   ├── lib.rs                # pymodule entry + global allocator
│   ├── rng.rs                # per-row RNG + the sampler drivers every plugin shells over
│   └── distributions/        # one Rust file per distribution
├── polars_stats/
│   ├── __init__.py           # public exports
│   ├── _lib.py               # plugin path resolution
│   └── distributions/
│       ├── _base.py          # ABCs + coercion / null helpers
│       └── _<name>.py        # one Python class per distribution
├── tests/
│   ├── distributions/<name>/ # one folder per distribution, one file per method
│   ├── property/             # hypothesis-based invariant tests
│   └── scipy_parity/         # scipy reference-oracle tests
├── benchmarks/               # internal benchmark harness (not part of the docs)
└── docs/                     # this documentation, one directory per Diataxis quadrant
```

## Stack

### Rust runtime

| Crate | Purpose |
|---|---|
| `polars` / `polars-arrow` | Series and expression types in Rust (pinned transitively by `pyo3-polars`) |
| `polars-core` | `POOL` and its `rayon` re-export, so the multi-draw fill runs on the thread pool Polars itself uses |
| `pyo3-polars` | the `#[polars_expr]` macro and FFI glue (source of ABI churn) |
| `pyo3` | Python FFI, abi3 for forward compatibility |
| `statrs` 0.19 | distribution math, and sampling except the binomial and uniform draws |
| `rand_distr` 0.6 | `O(1)`-amortised binomial draw (statrs' is `O(n)` per row); exact build version pinned by `Cargo.lock`, see [Design notes](explanation/design.md#binomial-sampling-uses-rand_distr-not-statrs) |
| `rand` 0.10 | `TryRng` / `SysRng` for the unseeded root seed |
| `rand_pcg` 0.10 | `Pcg64Mcg` per-row RNG for deterministic seeded sampling |
| `serde` | deserialise the static `seed` kwarg |

Deliberately excluded: `rand_chacha` (replaced by `rand_pcg`; per-row `ChaCha20`
construction made sampling markedly slower, see [Design notes](explanation/design.md#sampling-derives-a-fresh-per-row-rng-from-root_seed-row_index)),
`ndarray` (Polars is Arrow-native), `scirs2-stats` (pre-1.0). There is no direct `rayon` dependency either: the
multi-draw fill in `rng.rs` parallelises through the re-export in `polars-core`, on the same `POOL` Polars uses.

### Python runtime

Only `polars>=1.15`. No other runtime dependencies.

### Dev and CI tooling

* Dev dependencies are grouped in `pyproject.toml` (`testing`, `benchmarks`, `typing`, `docs`) and installed with
    `uv sync --group ...`.
* Tests run under `pytest` with `scipy` + `numpy` as the parity oracle (`tests/scipy_parity/`) and `hypothesis` for
    property tests; the `benchmarks/` comparison report measures wall-clock time and peak memory against `scipy.stats`.
* Python is checked by `ruff` (lint + format) and three type checkers in CI (`mypy`, `pyright`, `pyrefly`);
    Rust by `cargo fmt` (nightly) and `cargo clippy --all-features --all-targets -- -D warnings`.
    Prose and config are linted by `rumdl` (Markdown), `ryl` (YAML), `codespell`, `typos`, and `blacken-docs`, wired
    through `.pre-commit-config.yaml` and run by `prek` both locally and in CI, so the hook versions pinned there are
    the single source of truth. Do not bypass the hooks with `--no-verify` unless asked.

## Adding a distribution

Copy an existing pair, modify, repeat. Use `Uniform` (`_uniform.py` / `uniform.rs`) as the canonical continuous example
and `Bernoulli` as the discrete one. Each distribution opened up for contribution gets a GitHub issue carrying its
exact spec and checklist; that issue is canonical if it conflicts with this section.

1. **Rust.** Add `src/distributions/<name>.rs`, register it in `src/distributions/mod.rs`, and implement a
   `#[polars_expr]` only for methods that need it:

    * Always the samplers. Every one is a shell over a driver in `src/rng.rs`; none resolves a seed or writes a row
      loop itself, which is what keeps seeding, `null` propagation (a `null` in any input nulls the row) and the
      invalid-parameter error contract in one place. The per-row `<name>_sample` / `<name>_samples` (multi-draw,
      backing `samples`) take the parameter columns plus a row index as the last input: `<name>_sample` calls
      `sample_per_row_binary` (one parameter column) or `sample_per_row_ternary` (two), passing a `build` that
      validates and constructs the row's draw state; `<name>_samples` feeds `binary_param_rows` /
      `ternary_param_rows` into `samples_per_row`. The constant-parameter fast paths `<name>_sample_scalar` /
      `<name>_samples_scalar` are shells over `sample_by_index` / `samples_by_index`, taking
      `SampleScalarKwargs<<Name>ParamsKwargs>` / `SamplesScalarKwargs<<Name>ParamsKwargs>` so the parameters are
      validated once. All five drivers share the argument order `name, inputs, seed, [size], closures` and take the
      output dtype from what `draw` returns, so no call site names a polars type. The per-row drivers want the index
      pre-cast (`index.u64()?`, since `*_param_rows` hands back a borrowing iterator); the fast-path drivers take the
      raw `&inputs[0]` and cast it themselves. Give the distribution one named `fn draw` and call it from the
      per-row plugin and both shells; that shared call, not a test, is what keeps the three byte-identical.
    * When a method needs a **special function** (`erf`, log-gamma, regularized incomplete beta/gamma, ...) or has
      no elementary closed form: bind it in `statrs` (`pdf` / `pmf`, `cdf`, `ppf`, `ln_pdf` / `ln_pmf`, native `sf`,
      native `median`). Each shares one named `*_value` body between the per-row `value_keyed` helper (a hand-written
      shell over the `value_keyed_per_row` driver in `src/distributions/mod.rs`) and the constant-parameter
      `<name>_<method>_scalar` twin, so the two paths are byte-identical by construction. Write each twin as a
      one-line `#[polars_expr]` shell over `kwargs.value_keyed(&inputs[0], <method>_value)`, an inherent method on
      the distribution's `<Name>ParamsKwargs` struct that validates and builds the distribution once per call and
      then maps the body through the shared `value_keyed_scalar` driver. Putting the driver on the kwargs struct is
      what makes the hoist structural: a shell cannot build the distribution itself, so it cannot rebuild per row.
    * When a method is an **elementary** closed form (no special function: a Normal's
      `0.5*log(2*pi*e*sigma^2)`, the Uniform / Exponential / Cauchy `pdf` / `cdf` / `ppf`, `mean = n*p`, ...):
      **leave it in Python as a Polars expression, do not bind it in Rust.** A Rust binding for arithmetic Polars does
      natively is dead FFI surface. `Uniform` and `Bernoulli` are the canonical closed-form distributions: only
      `sample` and the validator touch Rust, everything else is a Polars hook in `_<name>.py`.
    * Factor the validating constructor into a `build_dist(...) -> PolarsResult<Dist>` helper so an invalid parameter
      maps through a `ComputeError` consistently.
2. **Python.** Add `polars_stats/distributions/_<name>.py`, subclassing `ContinuousDistribution` or
   `DiscreteDistribution`. In `__init__`, coerce each parameter with `coerce_param` / `coerce_n` (types only, **never
   validate values** at construction, let invalid values raise in Rust) and store the fast-path bundle
   `self._scalar_kwargs = scalar_kwargs(...)`. The base owns all routing (`sample`, `samples`, `_samples_columns`,
   `_value_plugin`, `_moment`); a subclass declares only what is distribution-specific:

    * `_plugin_prefix: ClassVar[str]` (e.g. `"normal"`), from which the base derives every sampler plugin name
      (`<prefix>_sample` / `_sample_scalar` / `_samples` / `_samples_scalar`), routing scalar vs column parameters off
      `_scalar_kwargs`.
    * `_param_exprs`, the coerced parameters as a tuple in plugin-input order; the *first* sets the output's root name.
    * The private formula hooks (`_pdf` / `_pmf`, `_cdf`, `_ppf`, and any `_log_*` / `_sf` closed form). For a
      statrs-backed method, return `self._value_plugin("<name>_<method>", value)`: the base routes constant parameters
      to the `_scalar` fast path and column parameters to the per-row plugin.
    * A validating plugin returning a reused quantity that raises on invalid parameters and nulls on a null one (e.g.
      `uniform_range` returns `max - min`). Write it as a `#[polars_expr]` shell over the generic
      `validate_params_binary` / `validate_params_unary` drivers in `src/distributions/mod.rs`: cast each input,
      take its accessor (`.f64()` / `.u64()`, so the two parameter dtypes may differ as Binomial's `(n, p)` does),
      and pass a closure that calls `build_dist` and returns the quantity to emit. Keep that closure a generic
      `F: Fn` so it monomorphises into the row loop. These drivers take no output name: polars resolves an
      expression's output name from its first input, so the column follows `inputs[0]` whatever the plugin calls
      its `Series`. For a statrs-backed
      distribution whose moment formulas may omit a parameter, expose the validator as `_checked_params` and gate
      every closed-form moment through `self._moment(<formula>)`; for a closed-form distribution whose validator
      is already part of every formula (`Uniform`, `Bernoulli`), weave it in directly and do not call `_moment`.

    **Never override the public `pdf` / `cdf` / ... methods**, nor the base-owned `sample` / `samples` /
    `_samples_columns` / `_value_plugin` / `_moment`. Export the class from `polars_stats/__init__.py`.
3. **Tests.** A new distribution touches its own files **and** several shared registries; missing a registry silently
   drops it from that suite (the run still passes, so nothing warns you). All of:

    * `tests/distributions/<name>/`: one file per method, mirroring `tests/distributions/bernoulli/`, including a
      `validation_test.py` asserting an invalid parameter raises `ComputeError` (not a null) for both scalar and column
      inputs.
    * `tests/scipy_parity/<name>_test.py`: one `Case` per method against `scipy.stats.<name>` through the shared
      `_harness.py` (default absolute tolerance `1e-12`, relaxed per `Case` to `1e-9` / `1e-6` for erf-based or
      binary-search-`ppf` methods).
    * Shared registries, one entry each: `tests/property/_specs.py` (`ALL_SPECS`: a `DistSpec` with
      `make` / `make_columns` / `make_masked` plus `density`, `eval_range`, and `integration_bounds` (continuous) or
      `support` (discrete), which drives the whole property suite), `tests/distributions/output_name_test.py` (the
      `sample` / `samples` output-naming contract), `tests/distributions/value_arg_str_test.py` (a `str` value
      argument means `pl.col(name)`), and, for distributions with the corresponding fast path,
      `tests/distributions/value_keyed_fast_path_test.py` and `tests/distributions/moment_fast_path_test.py`
      (scalar-vs-column validation contracts, including invalid-parameter cases).
4. **Update the umbrella issue** when the change merges.

## Numerical stability

**Every method must be accurate in the regime it exists to serve.** `log_sf` exists for the deep tail, so a `log_sf`
that returns `-inf` there does not work, even though every test passes. The base-class defaults `_cdf().log()`,
`_sf().log()` and `_isf(q) = _ppf(1 - q)` are a convenience, not an implementation: inheriting one is a decision to
justify. `_isf` is the sharpest case, because the loss happens *before* your code runs: `1 - q` resolves to `1.1e-16`
absolute, so the tail mass is already quantised to `1.1e-16 / q` relative and no inverse can recover it. Solve against
`q` itself, via a symmetry, a closed form, or entering a two-sided solve from the other end.

**A composed method inherits the weakest part's range, and the composition is often wider than the part.** `std()`
defaults to `variance().sqrt()`, and a variance that legitimately overflows can hide a standard deviation that does
not. Ask what the domain of the composed quantity is, not what the domain of its parts is.

**Reassociate before reaching for log space or a branch.** Computing a density as `exp(log_pdf)` fixes the subnormal
range but costs one to two orders of magnitude everywhere else, because `exp(log(rate))` does not round-trip to
`rate`, so it needs a threshold and `when/then/otherwise` evaluates both sides. `(rate * exp(t / 2)) * exp(t / 2)` is
the same product with no branch and no threshold constant. Rearrange so the single unavoidable rounding happens last.

**The test suite cannot catch this for you.** Three structural blind spots, each of which has produced a real defect:

* parity grids are finite and curated, so they never probe the extreme regime;
* `scipy` is not a valid oracle in the tails: its own `logcdf` / `logsf` are naive for the incomplete-beta and
    incomplete-gamma families, so parity *passes* while both libraries return `-inf`;
* property tests assert shape, not accuracy: monotonicity and mass-integrates-to-one are satisfied by a value that is
    relatively wrong by `1e-3`.

**Choose the oracle deliberately, and say which one you used and why.** `scipy` where it is finite and known accurate;
`mpmath` at high precision, or an exact closed-form identity, beyond that. Never assert against a `scipy` value that is
itself saturated. An oracle for a *discrete* inverse needs exact rational arithmetic, since a rounding there becomes a
jump between support points rather than a small error.

**Two acceptable outcomes, and no third.** Fix the algorithm, or document the caveat in
[Numerical accuracy](explanation/accuracy.md), quantified with a regime and a magnitude ("relative error
`~1.1e-16 / q`", not "may be inaccurate in the tails"). That page is the single home for accuracy caveats: no
per-class docstring blocks, no restating the same limit in several places. **A runtime warning is never the fix**: it
cannot fire per-row from inside the engine, and a Python-side scalar-only warning would break the scalar/column
symmetry the architecture guarantees.

**Claim a tolerance and justify it.** `1e-12` for elementary closed forms, `1e-10` for special-function methods
(`2e-10` through `erfc`, which is what `statrs` holds), `1e-9` for log-scale, `1e-8` for discrete log-mass and
support-sum entropy, `1e-6` or integer-valued for a discrete binary-search `ppf`. A relaxed tolerance needs a one-line
reason in the `Case`, not a shrug.

**Run `make audit` for a new distribution**, and add its oracle to the registry in `tools/accuracy_audit.py`. A
distribution absent from the audit is unaudited, exactly as one absent from `tests/property/_specs.py` is untested.
Never bound the sweep by what the implementation is known to be bad at: that is the defect's own shape used as a bound
on the instrument. Sweep extreme *parameters* too, not only extreme inputs.

**Ask what class a finding belongs to before closing it**, and treat "this method is exempt" as a hypothesis to probe
rather than an argument to accept. Both rules were bought the same way: a `pdf`-in-log-space defect fixed once and
found again elsewhere, and three distributions cleared from the `isf` fix by reasoning that each fell to the first
probe aimed at it.

**Where the recipes live.** `Exponential._log_sf` (an exact closed form), `Exponential._cdf` and `LogNormal.variance`
(the `sinh` identity standing in for the `expm1` Polars does not expose), `Exponential._log_cdf` and
`Uniform._log_cdf` (`log1p` on the near-certain side), `normal.rs`'s `ln_erfc` (a special function ported to log
space, the pattern for the hard cases), and the `isf_value` bodies in `normal.rs` (a symmetry) and `lognormal.rs`
(composing one).

## Conventions

Prose, in PRs and docs: no em dashes or double hyphens (use commas, colons, or parentheses); lead with the answer;
state uncertainty explicitly; end non-trivial answers with a short "Blind spots" section.

Code: KISS and YAGNI; production-grade type hints and explicit error handling; default to no comments, add one only when
the *why* is non-obvious. Do not refactor speculatively, do not introduce backwards-compatibility shims unless asked,
and do not add the `DistKind` dispatch macro (rejected for v1, see
[Design notes](explanation/design.md#one-rust-file-per-distribution-one-plugin-function-per-method-that-needs-rust)).

Tests assert on `pl.Series` / `pl.DataFrame` via `polars.testing`, not Python lists. For random output, assert the null
mask, not values. Genuinely scalar results are fine to read with `.item(...)` and compare via `pytest.approx`. scipy
parity stays on `numpy` `assert_allclose`.

Git: one change per distribution; commit messages mirror the existing history (`feat:`, `fix:`, `refactor:`); add a
co-authored-by trailer for AI-generated commits. Do not use destructive git operations without explicit approval.
