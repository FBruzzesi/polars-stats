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
`DiscreteDistribution`. The catalogue and the full table live in the [API reference](../reference/index.md#method-surface).

**Template-method split**: every value-keyed method is *concrete in the base*: it coerces the argument with `as_expr`,
applies `propagate_null_and_nan`, then delegates the maths to a private hook (`_pdf`, `_cdf`, `_ppf`, ...). **Subclasses
implement and override the `_x` hooks, never the public methods.** The hook receives an already-coerced `pl.Expr` and
returns the formula with no null or NaN handling; the base guarantees the input contract uniformly (null in, null out;
`NaN` in, `NaN` out, matching scipy) along with input coercion.

Composing defaults live in the base and call the other hooks:

* `_sf` defaults to `1 - _cdf(x)`
* `_isf` defaults to `_ppf(1 - q)`
* `_log_pdf` / `_log_pmf` / `_log_cdf` / `_log_sf` default to `.log()` of the underlying hook
* `median` defaults to `ppf(0.5)`, `std` to `variance().sqrt()`

A subclass overrides one of these only when `statrs` (or a closed form) is more numerically accurate, for instance
binding a native `ln_pdf` instead of letting `log_pdf` underflow in the tails.

## Column-valued parameters

Every distribution `__init__` coerces each parameter with a single shared helper, `coerce_param` (`coerce_n` for count
parameters like Binomial's `n`). The accepted inputs and their coercions are tabulated in
[Reference / Parameters and contracts](../reference/parameters-and-contracts.md#accepted-inputs).

A scalar is expanded with `pl.repeat(value, n=pl.len())`, a length-`N` expression rather than `pl.lit`, on purpose: the
plugin always receives a row-aligned
input, so it stays elementwise and `over` / `group_by` invoke it once per partition rather than as an aggregation.

On the Rust side, a plugin function receives `inputs: &[Series]` of `(value, param_1, ..., param_k)`, casts each to
`Float64Chunked`, iterates per chunk, propagates nulls, and constructs the `statrs` distribution per row. `kwargs`
carries only static config that cannot be column-valued: the sampler `seed`, plus the constant parameters in the
sampler fast path below (the one place a parameter rides in `kwargs`, valid precisely because there it is known to be a
scalar).

## Plugin granularity

**One Rust file per distribution, one `#[polars_expr]` per method that genuinely needs Rust**: trivial closed forms
(Bernoulli's `pmf`, `cdf`, `ppf`, moments; Uniform's `pdf`, `cdf`, ...) live in Python as `pl.Expr` and compose with the
expression engine. Methods that go through `statrs` (sampling, transcendental `pdf`/`pmf`, `cdf`, `inverse_cdf`, native
`ln_pdf`/`ln_pmf`, native `sf`) get a Rust plugin function.

**Parameter validation needs Rust.** A bare `pl.Expr` cannot raise per row, so any method computed in Python routes its
parameters through one small validating plugin that raises on a bad row and returns a reused quantity. For a
closed-form distribution that covers the whole surface: `Uniform.range` returns `max - min` and raises on `max <= min`,
`Bernoulli` validates `p` in `[0, 1]`, `Exponential` validates `rate > 0`. For a statrs-backed distribution it covers
the closed-form moments only, since the value-keyed methods already validate inside their own plugin: `normal_sigma`
validates `sigma > 0`, so `Normal.mean()` raises exactly as `Normal.pdf()` does. Either way a pure-math `mean` makes
one FFI round-trip and reports an invalid parameterisation consistently, trading a little throughput for a uniform
error surface.

## Sampling

`sample(seed)` returns one draw per row from the row-specific distribution; `samples(size, seed)` returns an
`Array(inner=..., shape=size)`, drawn as one native multi-draw plugin call per frame: row `i`'s `size` draws are
consecutive values from the one per-row stream keyed `(seed, i)`, so `samples(size=1)` matches `sample` bit for bit
and growing `size` extends each row's array without changing existing draws.

**Per-row seeding**: the root seed is resolved once per plugin call (`OsRng` when `seed=None`), then each row derives
its own `Pcg64Mcg` generator from `(root_seed, row_index)` via two splitmix64 mixing draws. The row index arrives as a
regular input column (`pl.int_range(0, pl.len())`), so it tracks the partition under `over` / `group_by`.

Identical `(root_seed, row_index)` always yields an identical stream, which is exactly what makes `sample`
elementwise and invariant to chunking and thread count. `Pcg64Mcg` is cheap to construct (a handful of integer ops, no
key schedule), passes TestU01 BigCrush, and is stable across `rand_pcg` releases and platforms, so seeded results are
reproducible across OS and architecture.

**Constant-parameter fast path**: when every distribution parameter is a Python scalar (the common case), the sampler
takes a dedicated plugin, `<name>_sample_scalar`. The parameters travel in `kwargs` and are validated once; only the row
index crosses FFI, instead of one full-length `pl.repeat` column per parameter that the general plugin would marshal and
re-validate on every row. The shared `sample_by_index` helper in `rng.rs` resolves the seed once and maps the dense,
non-null index straight into the typed output. Each distribution declares its fast path through the
`sample_scalar_plugin!` macro in `rng.rs` (kwargs struct, output dtype, one-time `build`, per-row `draw`), which
generates the kwargs struct and the plugin function, so a new distribution cannot drift from the pattern. It reuses
the same `(root_seed, row_index)` seeding and the same draw as the per-row path, so output is byte-identical for the
same seed (a property test pins that equality); column-valued parameters still take the general per-row plugin.

!!! info "Earlier `ChaCha20` design (removed)"

    A previous design advanced a single `ChaCha20Rng` once per row in iteration order, which coupled rows across chunks
    and was not streaming-safe. The naive fix, constructing a `ChaCha20Rng` per row, made sampling markedly slower (a
    key schedule plus a keystream block per draw). Per-row `Pcg64Mcg` is both correct and cheap and replaced it;
    `statrs`'s sampling traits consume it directly.

## Null and error contract

The full table is in
[Reference / Parameters and contracts](../reference/parameters-and-contracts.md#nulls-nans-and-errors). An invalid
parameter value raises a `ComputeError` and fails the evaluation; `null` is reserved for `null` *inputs*.
Construction rejects only wrong *types*. There is no early Python validation of parameter values, so a bad scalar and a
bad column row surface identically.

## Stack

The math runs on `statrs` 0.18; the plugin glue is `pyo3-polars` over `pyo3` (abi3); per-row seeded RNG is `rand_pcg`
(`Pcg64Mcg`), with `rand` for `OsRng`; `serde` deserialises the static `seed` kwarg. The full dependency rationale and
the deliberately-excluded crates are in [Contributing / Stack](../contributing.md#stack), and the repository layout is
in [Contributing / Repository layout](../contributing.md#repository-layout). Supported Python, Polars, and OS versions
are in [Reference / Compatibility](../reference/index.md#compatibility).
