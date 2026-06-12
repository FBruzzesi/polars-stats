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
make lint       # ruff + rumdl + cargo fmt (nightly) + clippy
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
| `rand_distr` 0.4 | `O(1)`-amortised binomial draw (statrs' is `O(n)` per row); exact build version pinned by `Cargo.lock`, see [Design notes](design.md#binomial-sampling-uses-rand_distr-not-statrs) |
| `rand` 0.8 | `RngCore` / `OsRng` for the unseeded root seed |
| `rand_pcg` 0.3 | `Pcg64Mcg` per-row RNG for deterministic seeded sampling |
| `serde` | deserialise the static `seed` kwarg |

Deliberately excluded: `rand_chacha` (replaced by `rand_pcg`; per-row `ChaCha20`
construction regressed sampling 10 to 20x, see [Design notes](design.md#sampling-derives-a-fresh-per-row-rng-from-root_seed-row_index)),
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
    through `.pre-commit-config.yaml`. Do not bypass the hooks with `--no-verify` unless asked.

## Adding a distribution

Copy an existing pair, modify, repeat. Use `Uniform` (`_uniform.py` / `uniform.rs`) as the canonical continuous example
and `Bernoulli` as the discrete one. The issue tracker has one file per distribution with the exact checklist; the issue
is canonical if it conflicts with this section.

1. **Rust.** Add `src/distributions/<name>.rs`, register it in `src/distributions/mod.rs`, and implement a
   `#[polars_expr]` only for methods that need it:

    * Always `sample` (native `statrs::Distribution::sample`). The sampler takes a per-row index as its last input
      `Series` and derives its RNG from `src/rng.rs`: `let rngs = kwargs.row_rngs();` once outside the loop, then
      `rngs.rng(i)` inside. Never reseed from chunk position. Use `try_*_elementwise` so a `null` in any input
      propagates and an invalid parameter raises.
    * When `statrs` is the cheapest path: `pdf` / `pmf`, `cdf`, `ppf`, `ln_pdf` / `ln_pmf`, native `sf`, native
      `median`.
    * When the closed form is trivial: leave it in Python instead.
    * Factor the validating constructor into a `build_dist(...) -> PolarsResult<Dist>` helper so an invalid parameter
      maps through a `ComputeError` consistently.
2. **Python.** Add `polars_stats/distributions/_<name>.py`, subclassing `ContinuousDistribution` or
   `DiscreteDistribution`. Coerce each parameter with `coerce_param(value, name=...)`. **Do not validate parameter
   *values* at construction**, coerce types only and let invalid values raise in Rust. Implement the private formula
   hooks (`_pdf` / `_pmf`, `_cdf`, `_ppf`, and any `_log_*` / `_sf` closed form), plus the
   `_samples_scalar_plugin` class var and `_samples_columns`. So closed-form methods raise on invalid params too,
   route them through a small validating plugin that returns a reused quantity (e.g. `uniform_range`).
   **Never override the public `pdf` / `cdf` / ... methods**: the base owns them. Export the class from
   `polars_stats/__init__.py`.
3. **Tests.** Two homes: `tests/distributions/<name>/`, one file per method (including a `validation_test.py` asserting
   an invalid parameter raises `ComputeError` for both scalar and column inputs); and `tests/scipy_parity/<name>_test.py`
   against `scipy.stats.<name>` to within `1e-10` (closed-form) or `1e-6` (binary-search `ppf`), compared with
   `np.testing.assert_allclose`.
4. **Update the umbrella issue** when the change merges.

## Conventions

Prose, in PRs and docs: no em dashes or double hyphens (use commas, colons, or parentheses); lead with the answer;
state uncertainty explicitly; end non-trivial answers with a short "Blind spots" section.

Code: KISS and YAGNI; production-grade type hints and explicit error handling; default to no comments, add one only when
the *why* is non-obvious. Do not refactor speculatively, do not introduce backwards-compatibility shims unless asked,
and do not add the `DistKind` dispatch macro (rejected for v1, see
[Design notes](design.md#one-rust-file-per-distribution-one-plugin-function-per-method-that-needs-rust)).

Tests assert on `pl.Series` / `pl.DataFrame` via `polars.testing`, not Python lists. For random output, assert the null
mask, not values. Genuinely scalar results are fine to read with `.item(...)` and compare via `pytest.approx`. scipy
parity stays on `numpy` `assert_allclose`.

Git: one change per distribution; commit messages mirror the existing history (`feat:`, `fix:`, `refactor:`); add a
co-authored-by trailer for AI-generated commits. Do not use destructive git operations without explicit approval.
