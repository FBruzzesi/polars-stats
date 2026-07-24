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

To preview these docs locally:

```bash
uv run --group docs zensical serve
```

## Stack

### Rust runtime

| Crate | Purpose |
|---|---|
| `polars` / `polars-arrow` | Series and expression types in Rust (pinned transitively by `pyo3-polars`) |
| `pyo3-polars` | the `#[polars_expr]` macro and FFI glue (source of ABI churn) |
| `pyo3` | Python FFI, abi3 for forward compatibility |
| `statrs` 0.18 | distribution math, and sampling except the binomial draw |
| `rand_distr` 0.4 | `O(1)`-amortised binomial draw (statrs' is `O(n)` per row); exact build version pinned by `Cargo.lock`, see [Design notes](explanation/design.md#binomial-sampling-uses-rand_distr-not-statrs) |
| `rand` 0.8 | `RngCore` / `OsRng` for the unseeded root seed |
| `rand_pcg` 0.3 | `Pcg64Mcg` per-row RNG for deterministic seeded sampling |
| `serde` | deserialise the static `seed` kwarg |

Deliberately excluded: `rand_chacha` (replaced by `rand_pcg`; per-row `ChaCha20`
construction made sampling markedly slower, see [Design notes](explanation/design.md#sampling-derives-a-fresh-per-row-rng-from-root_seed-row_index)),
`ndarray` (Polars is Arrow-native), `rayon` (Polars parallelises at the planner), `scirs2-stats` (pre-1.0).

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
and `Bernoulli` as the discrete one. The issue tracker has one file per distribution with the exact checklist; the issue
is canonical if it conflicts with this section.

1. **Rust.** Add `src/distributions/<name>.rs`, register it in `src/distributions/mod.rs`, and implement a
   `#[polars_expr]` only for methods that need it:

    * Always the samplers, deriving the RNG from `src/rng.rs` (`let rngs = kwargs.row_rngs();` once outside the loop,
      then `rngs.rng(i)` inside; never reseed from chunk position; `try_*_elementwise` so a `null` in any input
      propagates and an invalid parameter raises). The per-row `<name>_sample` / `<name>_samples` (multi-draw, backing
      `samples`) take the parameter columns plus a row index as the last input; the constant-parameter fast paths
      `<name>_sample_scalar` / `<name>_samples_scalar` (parameters validated once in `kwargs`) come from the
      `sample_scalar_plugin!` macro in `src/rng.rs`, generated from one shared `build` / `draw` so they stay
      byte-identical to the per-row path.
    * When a method needs a **special function** (`erf`, log-gamma, regularized incomplete beta/gamma, ...) or has
      no elementary closed form: bind it in `statrs` (`pdf` / `pmf`, `cdf`, `ppf`, `ln_pdf` / `ln_pmf`, native `sf`,
      native `median`). Each shares one named `*_value` body between the per-row `value_keyed` helper (from the
      `value_keyed_per_row!` macro) and the constant-parameter `<name>_<method>_scalar` twin (from the
      `value_keyed_scalar_plugins!` macro), both over the shared `value_keyed_scalar` driver, so the two paths are
      byte-identical by construction. All three macros live in `src/distributions/mod.rs`.
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
      `uniform_range` returns `max - min`). The two-parameter validators are generated by the `param_validator!` macro
      (`src/distributions/mod.rs`): declare each `<name>: <dtype> => <accessor>`, the `build`, the `returns` expression,
      and `output_name = inputs[i]`. One-parameter validators (`bernoulli_proba`, `exponential_rate`) use the macro's
      unary arm (a single `<name>: <dtype> => <accessor>`). For a statrs-backed distribution
      whose moment formulas may omit a parameter, expose the validator as `_checked_params` and gate every closed-form
      moment through `self._moment(<formula>)`; for a closed-form distribution whose validator is already part of every
      formula (`Uniform`, `Bernoulli`), weave it in directly and do not call `_moment`.

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
