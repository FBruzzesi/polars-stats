---
icon: lucide/lightbulb
---

# Design notes

The *why* behind the choices in [Architecture](architecture.md), and the questions still open. If you want to know what
the code does, read [Architecture](architecture.md). If you want to know why it does it that way, read this.

## Decisions taken

### Column-valued parameters travel in `inputs`, not `kwargs`

Polars plugins receive `inputs: &[Series]` (lazy, length-matched) and `kwargs` (static, JSON-serialised at planning
time). Putting parameters in `kwargs` would block column-valued parameters, which is the whole differentiator. So every
plugin call passes `(value, param_1, ..., param_k)` as `inputs`, and `kwargs` carries only static config (`seed`).

### One Rust file per distribution, one plugin function per method that needs Rust

Three options were considered: (A) one function per `(distribution, method)`; (B) one function per method with an
internal `DistKind` enum dispatch and a row-level `match`; (C) Rust only where a Polars expression cannot express the
method, Python otherwise.

The result is **A + C**. Closed-form methods sit happily in Python as `pl.Expr`; only methods that go through `statrs`
get a Rust plugin function. Option B was rejected: the enum dispatch and a code-generation macro add a build-time DSL
and a row-level `match` cost for no clear win at 17 distributions. Hand-written per-distribution files stay clearer;
revisit only past ~25 distributions.

### Sampling derives a fresh per-row RNG from `(root_seed, row_index)`

Every sampler needs one property: a deterministic, independent stream per row that depends only on
`(root_seed, row_index)`, never on position within a chunk. That makes output invariant to how Polars chunks or threads
the input, so `sample` is genuinely elementwise.

The generator is `Pcg64Mcg` (`rand_pcg`): cheap construction (no key schedule), passes TestU01 BigCrush, output stable
across releases and platforms. The root seed is resolved once per call (`OsRng` when `seed=None`), then each row's
128-bit state is mixed from `(root_seed, row_index)` with two splitmix64 draws.

This replaced an earlier "single `ChaCha20Rng` advanced once per row in iteration order" design, which coupled rows
across chunks (order-dependent, not streaming-safe). The naive fix, a `ChaCha20Rng` per row, regressed sampling 10 to
20x. A one-shot hash-to-uniform would be cheaper still but only serves distributions needing a single uniform per draw,
so it is deliberately not the foundation.

### Invalid parameters raise, they never silently null

An invalid parameter value, scalar or one bad column row (`std_dev <= 0`, `max <= min`, `p` outside `[0, 1]`, a
non-finite bound), maps the `statrs` constructor error through a `ComputeError` and fails the whole evaluation.

This reverses an earlier "produce null, keep the pipeline running" decision. Silently nulling hides a modelling error: a
user who does not check for nulls gets wrong answers downstream, and an invalid-parameter null is indistinguishable from
a legitimately-null input. Raising is loud, uniform across distributions, and uniform across scalar vs column inputs,
because scalars are coerced to columns and validated per row exactly like columns. Construction rejects only wrong
*types*. A closed-form distribution cannot raise from a bare `pl.Expr`, so it routes parameters through one small
validating plugin (see [Architecture / Plugin granularity](architecture.md#plugin-granularity)). A `strict=False` opt-in
that nulls instead of raising is deferred until a user asks.

### `NotImplementedError` for permanently undefined moments, `null` for regime-dependent

Cauchy's mean and variance are permanently undefined, so Cauchy raises `NotImplementedError`: silently returning null
would hide a modelling error from a user who chains `.mean().sum()`. Student-t with `df <= 1` also has no mean, but the
same distribution with `df > 1` does. That is regime-dependent, and a user sweeping a parameter across the threshold
should not get an exception that breaks the sweep, so it returns `null`. Inconsistent on its face, defensible per case,
documented in each class.

## Open questions

- **Parameter naming.** Classes name parameters after each distribution's conventional parameters
  (`Normal(mean, std_dev)`, `Uniform(min, max)`, `Exp(rate)` rather than `Exp(scale)`), which avoids the `scipy`
  `(loc, scale)` collision across location-scale families but breaks scipy-compat at the constructor. Each docstring
  spells out the `scipy` reparameterisation. Whether to keep this or adopt `scipy`'s uniform `(loc, scale)` is not yet
  settled; revisit before `1.0`.
- **API namespace.** `polars_stats.Normal(...).pdf(x)` (scipy-style, current) versus
  `pl.col("x").dist.normal_pdf(loc=..., scale=...)` (Polars namespace style, fits `.dt` / `.str` / `.list`). Picked
  scipy-style without strong evidence; revisit after first external feedback.
- **Sample output dtype.** Currently per distribution (`Boolean`, `UInt64`, `Float64`); `scipy` returns `f64`
  uniformly. Revisit if it forces awkward casting in user code.
- **Streaming-safe sampling.** The per-row keying already removed cross-chunk coupling; what is unverified is the
  streaming engine's handling of the injected row-index expression. Add a streaming test, then relax the `.collect()`
  guidance.
- **GIL release inside plugin entry points.** Wrapping the hot loop in `py.allow_threads` is a hardening pass tracked
  before v0.5, not yet audited.
- **Pure-Rust consumption.** Publishing to crates.io works, but the `cdylib` + PyO3 stack makes Rust-only use awkward.
  Splitting into `polars-stats-core` (no PyO3) + `polars-stats` (binding) is the v2 fix.

## Risks

| Risk | Mitigation |
|---|---|
| `pyo3-polars` breaks its ABI on most Polars minor releases | Automated bump + CI on a weekly cadence; pin the upper bound to the next Polars minor |
| `statrs` is a single-maintainer project | Track activity; vendor or fork the specific distributions we use if upstream archives |
| `statrs` accuracy in the tails (binary-search `inverse_cdf` for Gamma / Poisson / NegBinom / ...) | Per-distribution tolerance documented in tests; v2 candidate: native `ppf` via the `special` crate or Newton refinement |
| `rand` 0.8 to 0.9 transition (`distributions` renamed to `distr`) | Pin 0.8 until `statrs` moves; follow in lockstep |
| Reference values for tests come from `scipy.stats`, the library we partly replace | Acknowledged; freeze the SciPy version in dev deps |
| abi3 compatibility across the Python range | Test on all supported versions in CI even though abi3 claims compatibility |

## Blind spots

- **Documentation drift.** Treat any change to plugin granularity, the null contract, or seeding as a doc-update change.
- **Per-distribution Rust files scale linearly.** ~16 remaining distributions x ~5 Rust methods each. If that becomes a
  build-time or binary-size problem, `DistKind` dispatch is the way back.
- **Per-row RNG cost.** Constructing a fresh `Pcg64Mcg` per draw is more work than advancing one shared RNG. It is what
  buys chunk- and thread-invariance; the benchmarks guard against a regression like the per-row-`ChaCha20` one, not
  against this baseline cost itself.
- **Raise-on-invalid-row is untested at scale.** A pipeline with a million bad rows raises on the first; the message
  names the offending value but not the row index, which is hard to locate in a wide frame.
- **`scipy` parity tests create a circular dev dependency** on the library being partly replaced.
- **MSRV is not pinned.** `rust-toolchain.toml` pins a nightly (needed for the `rustfmt.toml` import-granularity
  options). Decide a stable N-2 floor before release.
- **`statrs` lacks some `scipy` distributions** (GEV, Skew Normal, Truncated Normal, mixtures). If users need these we
  either upstream to `statrs` or fork.
- **Demand for column-valued parameters is unvalidated** beyond the authors' satellite-telemetry anomaly-detection case.
  Check Polars Discord / GitHub before v0.5.
