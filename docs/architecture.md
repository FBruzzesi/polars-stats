---
icon: lucide/layers
---

# Architecture

The canonical "what `polars-stats` is and how it is wired". For the rationale behind these choices and the open
questions, see [Design notes](design.md).

## Three layers

* Python API layer (`polars_stats/`): Distribution classes (Normal, Bernoulli, ...). Coerce params into row-aligned
    `pl.Expr`, register plugin calls. Implement methods directly as `pl.Expr` when the closed form is trivial.

* FFI layer (`pyo3-polars` plugin functions). One `#[polars_expr]` per (distribution, method) where Rust is required.
    Per-row null propagation; an invalid parameter on a row raises.

* Math layer (`statrs`): Trusted upstream for pdf, cdf, ppf, moments, sampling

## Public method surface

The method surface (`pdf`/`pmf`, `cdf`, `sf`, `ppf`, `isf`, the `log_*` family, `mean`, `variance`, `std`, `median`,
`entropy`, `sample`, `samples`) is defined on the abstract base classes `ContinuousDistribution` and
`DiscreteDistribution`. The catalogue and the full table live in [Distributions](distributions.md#method-surface).

**Template-method split**: every value-keyed method is *concrete in the base*: it coerces the argument with `as_expr`,
applies `propagate_null`, then delegates the maths to a private hook (`_pdf`, `_cdf`, `_ppf`, ...). **Subclasses
implement and override the `_x` hooks, never the public methods.** The hook receives an already-coerced `pl.Expr` and
returns the formula with no null handling; the base guarantees the null contract and input coercion uniformly.

Composing defaults live in the base and call the other hooks:

* `_sf` defaults to `1 - _cdf(x)`
* `_isf` defaults to `_ppf(1 - q)`
* `_log_pdf` / `_log_pmf` / `_log_cdf` / `_log_sf` default to `.log()` of the underlying hook
* `median` defaults to `ppf(0.5)`, `std` to `variance().sqrt()`

A subclass overrides one of these only when `statrs` (or a closed form) is more numerically accurate, for instance
binding a native `ln_pdf` instead of letting `log_pdf` underflow in the tails.

## Column-valued parameters

The differentiating feature. Every distribution `__init__` coerces each parameter with a single shared helper,
`coerce_param`:

| Input | Coercion |
|---|---|
| `float`, `int` | `pl.repeat(value, n=pl.len(), dtype=Float64)` (row-aligned, keeps `is_elementwise=True` valid under `over` / `group_by`) |
| `pl.Expr` | passed through |
| `pl.Series` | `pl.lit(series)` |
| `str` | `pl.col(name)` |
| other | `TypeError` |

A scalar is expanded to a length-`N` expression (not `pl.lit`) on purpose: the plugin always receives a row-aligned
input, so it stays elementwise and `over` / `group_by` invoke it once per partition rather than as an aggregation.

On the Rust side, a plugin function receives `inputs: &[Series]` of `(value, param_1, ..., param_k)`, casts each to
`Float64Chunked`, iterates per chunk, propagates nulls, and constructs the `statrs` distribution per row. `kwargs`
carries only static config that cannot be column-valued, namely the sampler `seed`.

## Plugin granularity

**One Rust file per distribution, one `#[polars_expr]` per method that genuinely needs Rust**: trivial closed forms
(Bernoulli's `pmf`, `cdf`, `ppf`, moments; Uniform's `pdf`, `cdf`, ...) live in Python as `pl.Expr` and compose with the
expression engine. Methods that go through `statrs` (sampling, transcendental `pdf`/`pmf`, `cdf`, `inverse_cdf`, native
`ln_pdf`/`ln_pmf`, native `sf`) get a Rust plugin function.

**Parameter validation needs Rust.** A bare `pl.Expr` cannot raise per row, so to enforce the invalid-parameter contract
a closed-form distribution routes its parameters through one small validating plugin that raises on a bad row and
returns a reused quantity. `Uniform.range` returns `max - min` and raises on `max <= min`; `Bernoulli` validates `p` in
`[0, 1]`; `Normal` validates `std_dev > 0`. Every closed-form method derives from that quantity, so even a pure-math
`mean` performs one FFI round-trip and reports an invalid parameterisation consistently. This trades a little throughput
for a uniform error surface.

## Sampling

`sample(seed)` returns one draw per row from the row-specific distribution; `samples(size, seed)` returns an
`Array(inner=..., shape=size)`, implemented in the base by calling `sample` `size` times with derived sub-seeds.

**Per-row seeding**: the root seed is resolved once per plugin call (`OsRng` when `seed=None`), then each row derives
its own `Pcg64Mcg` generator from `(root_seed, row_index)` via two splitmix64 mixing draws. The row index arrives as a
regular input column (`pl.int_range(0, pl.len())`), so it tracks the partition under `over` / `group_by`.

Identical `(root_seed, row_index)` always yields an identical stream, which is exactly what makes `sample`
elementwise and invariant to chunking and thread count. `Pcg64Mcg` is cheap to construct (a handful of integer ops, no
key schedule), passes TestU01 BigCrush, and is stable across `rand_pcg` releases and platforms, so seeded results are
reproducible across OS and architecture.

!!! info "Earlier `ChaCha20` design (removed)"

    A previous design advanced a single `ChaCha20Rng` once per row in iteration order, which coupled rows across chunks
    and was not streaming-safe. The naive fix, constructing a `ChaCha20Rng` per row, regressed sampling 10 to 20x (a key
    schedule plus a keystream block per draw). Per-row `Pcg64Mcg` is both correct and cheap and replaced it; `statrs`'s
    sampling traits consume it directly.

## Null and error contract

The full table is in [User guide / Nulls and errors](user-guide/nulls-and-errors.md). The rule: an invalid parameter value
raises a `ComputeError` and fails the evaluation; `null` is reserved for `null` *inputs*. Construction rejects only
wrong *types*. There is no early Python validation of parameter values, so a bad scalar and a bad column row surface
identically.

## Repository layout

```text
polars-stats/
├── Cargo.toml
├── pyproject.toml
├── rust-toolchain.toml
├── src/
│   ├── lib.rs                # pymodule entry + global allocator
│   ├── rng.rs                # shared per-row RNG (SampleKwargs, RowRngs)
│   └── distributions/        # one Rust file per distribution
├── polars_stats/
│   ├── __init__.py           # public exports
│   ├── _lib.py               # plugin path resolution
│   └── distributions/
│       ├── _base.py          # ABCs + coercion / null helpers
│       └── _<name>.py        # one Python class per distribution
├── tests/
│   ├── distributions/<name>/ # one folder per distribution, one file per method
│   └── scipy_parity/         # scipy reference-oracle tests
├── benchmarks/               # pytest-benchmark sampler hot-path guards
└── docs/                     # this documentation
```

## Stack

The math runs on `statrs` 0.18; the plugin glue is `pyo3-polars` over `pyo3` (abi3); per-row seeded RNG is `rand_pcg`
(`Pcg64Mcg`), with `rand` for `OsRng`; `serde` deserialises the static `seed` kwarg. The full dependency rationale and
the deliberately-excluded crates are in [Contributing / Stack](contributing.md#stack).

## Compatibility and release

| Dimension | Values |
|---|---|
| OS | Target: Linux x86_64/aarch64, macOS arm64/x86_64, Windows x86_64. PR CI runs Linux x86_64 only; the full matrix is a release-build concern. |
| Python | 3.10 to 3.14, single abi3 wheel. |
| Polars | `>=1.15`; PR CI tests the oldest pin (`1.15.0`) and the latest. |

`pyo3-polars` breaks its ABI on most Polars minor releases, so the release cadence is a patch within roughly one week of
each Polars minor. License: MIT OR Apache-2.0.
