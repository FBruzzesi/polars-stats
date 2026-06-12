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
plugin call passes `(value, param_1, ..., param_k)` as `inputs`, and `kwargs` carries only static config (`seed`). The
one exception is the constant-parameter sampler fast path below, which passes scalars in `kwargs` precisely because they
are known not to be column-valued there.

### One Rust file per distribution, one plugin function per method that needs Rust

Three options were considered:

1. one function per `(distribution, method)`;
2. one function per method with an internal `DistKind` enum dispatch and a row-level `match`;
3. Rust only where a Polars expression cannot express the method, Python otherwise.

The result is **1 & 3**. Closed-form methods sit happily in Python as `pl.Expr`; only methods that go through `statrs`
get a Rust plugin function.

### Sampling derives a fresh per-row RNG from `(root_seed, row_index)`

Every sampler needs one property: a deterministic, independent stream per row that depends only on
`(root_seed, row_index)`, never on position within a chunk. That makes output invariant to how Polars chunks or threads
the input, so `sample` is genuinely elementwise.

The generator is `Pcg64Mcg` (`rand_pcg`): cheap to construct (no key schedule), good statistical quality, and output
stable across releases and platforms, so seeded results stay reproducible. The root seed is resolved once per call
(`OsRng` when `seed=None`); each row then derives its own generator from `(root_seed, row_index)`.

This replaced an earlier "single `ChaCha20Rng` advanced once per row in iteration order" design, which coupled rows
across chunks (order-dependent, not streaming-safe). The naive fix, a `ChaCha20Rng` per row, regressed sampling 10 to
20x. A one-shot hash-to-uniform would be cheaper still but only serves distributions needing a single uniform per draw,
so it is deliberately not the foundation.

### Constant parameters take a sampler fast path

`sample` ships a second plugin, `<name>_sample_scalar`, used when every parameter is a Python scalar. The general
sampler is built for the differentiator (column-valued parameters), but it makes the common constant-parameter case pay
for machinery it does not use: each scalar is expanded to a full-length `pl.repeat` column, marshalled across FFI, and
re-validated on every row, and the distribution is rebuilt per row. For a cheap draw (uniform is one multiply-add) that
fixed overhead dominates, leaving the sampler slower than scipy until well past 100k rows.

The fast path passes the constant parameters in `kwargs`, validates and builds the distribution once, and sends only the
row index as an input. It keeps the exact `(root_seed, row_index)` seeding and the same draw, so its output is
byte-identical to the per-row path for any seed; that equality is the contract, pinned by a property test
(`test_sample_scalar_fast_path_matches_per_row`) rather than left implicit. The result is a 2 to 9x speedup over scipy
at 100k+ rows across distributions, and lower peak memory (no constant columns), with reproducibility and the chunk- and
thread-invariance guarantees untouched.

It is the one deliberate exception to "parameters travel in `inputs`, not `kwargs`": admissible precisely because the
path is selected only when the parameters are known scalars, so nothing column-valued is ever forced into `kwargs`.
`sample_iter` was rejected for the loop body: it advances a single stream in row order, which couples rows across chunks
and breaks the invariance guarantee the per-row seeding exists to provide.

`samples` does, however, use one stream *per row*: row `i`'s `size` draws are consecutive values from the
`(root_seed, i)` stream, the same stream `sample` takes its single draw from. That stays chunk-invariant because the
stream is keyed by global row position; what remains rejected is any stream shared across rows. The per-row stream is
what makes `samples(size=1)` equal `sample` bit for bit and `samples` prefix-stable in `size` (both pinned by property
tests). The measured trade-off at 100k rows, `size=20`: cheap-draw distributions speed up (no per-draw re-seeding)
while the normal slows ~20% (consecutive draws are sequentially dependent, so the ziggurat loses cross-draw
instruction overlap); recorded in CHANGELOG.md as a pre-release seeded-output change.

### Binomial sampling uses `rand_distr`, not `statrs`

`statrs` 0.18 implements `Distribution<u64> for Binomial` as `(0..n).fold(...)`, one uniform draw per trial, so a sampled
row costs `n` RNG draws. At `n = 10_000` that is 10,000 uniforms per row, turning the sampler `O(n)`; the constant-factor
wins over scipy held only for small `n`. The binomial sampler therefore draws from `rand_distr::Binomial` (inversion for
small `n*p`, BTPE otherwise, both `O(1)`-amortised), keeping sampling time flat in `n`. This is the one place sampling
does not go through `statrs`; every value-keyed method (`pmf`, `cdf`, `ppf`, ...) still builds the `statrs` distribution.

### Invalid parameters raise, they never silently null

An invalid parameter value, scalar or one bad column row (`std_dev <= 0`, `max <= min`, `p` outside `[0, 1]`, a
non-finite bound), maps the `statrs` constructor error through a `ComputeError` and fails the whole evaluation.

This reverses an earlier "produce null, keep the pipeline running" decision. Silently nulling hides a modelling error: a
user who does not check for nulls gets wrong answers downstream, and an invalid-parameter null is indistinguishable from
a legitimately-null input. Raising is loud, uniform across distributions, and uniform across scalar vs column inputs,
because scalars are coerced to columns and validated per row exactly like columns. Construction rejects only wrong
*types*. A closed-form distribution cannot raise from a bare `pl.Expr`, so it routes parameters through one small
validating plugin (see [Architecture / Plugin granularity](architecture.md#plugin-granularity)).

### Moments that are undefined

Every distribution shipped today has finite moments on its valid parameter range, so this policy does not bite yet; it
governs distributions on the roadmap. Two cases, handled differently on purpose:

* **Permanently undefined** (e.g. a Cauchy mean): raise `NotImplementedError`. Silently returning null would hide a
  modelling error from a user who chains `.mean().sum()`.
* **Undefined only in part of the parameter range** (e.g. a Student-t mean with `df <= 1`, defined for `df > 1`):
  return `null`. A user sweeping a parameter across the threshold should not get an exception that breaks the sweep.

Inconsistent on its face, defensible per case, and to be documented in each class as it lands.
