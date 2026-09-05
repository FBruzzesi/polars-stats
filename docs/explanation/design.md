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

The result is **1 & 3**. Rust gets the RNG, the special functions (`erf`, log-gamma, the regularized incomplete beta
or gamma), and anything with no elementary closed form. A Rust binding for arithmetic Polars does natively is dead FFI
surface: more to compile, one more plugin name to keep literal, one more path to hold in parity with its own fast-path
twin, and no accuracy or speed gain for it.

Being expressible in Polars is necessary for staying in Python, but not sufficient. The second test is **whether every
row reaches the validator**: a `pl.Expr` cannot raise per row, so a method that must reject an invalid parameterisation
depends on the validating plugin actually running on the offending row. From polars 1.44 a `when` / `then` /
`otherwise` **arm** is masked to null on the rows it does not select, so a validator reachable only from inside an arm
never sees them and the method silently returns a value
([pola-rs/polars#29005](https://github.com/pola-rs/polars/issues/29005)). The **condition** is not masked, and an
unbranched expression has nothing to mask. So a closed form stays in Python when its validated parameter is read
unconditionally or named in a `when(...)` condition, and moves to Rust when it is read only inside an arm, which is
what branching on the evaluation value always looks like.

The table below is the rule, which every new distribution follows. It is not yet a description of the tree:
`Geometric` and `Uniform` still assemble their value-keyed methods in Polars and are being ported one at a time; the
[reference index](../reference/index.md) carries the resulting known limitation. Method by method:

| Method | In Rust? | Notes |
|---|---|---|
| `sample` / `samples` | **always** | Not derivable in Python. Usually `statrs`' own `Distribution::sample`. |
| parameter validation | **always** | One validation-only round-trip per distribution, since a bare `pl.Expr` cannot raise per row. |
| `pdf` / `pmf`, `cdf` | **always** | Native `Continuous::pdf` / `Discrete::pmf`, `*CDF::cdf` where `statrs` has them; a hand-written Rust body where the closed form is elementary. Branching on the value puts the validator inside an arm, so these cannot stay in Polars. |
| `ppf` | **always** | `*CDF::inverse_cdf`, a closed form for some families and a binary search for others. Which one sets the parity tolerance. Elementary inverses are hand-written Rust bodies, for the same arm-masking reason. |
| `log_pdf` / `log_pmf` | **always**, override the default | The base default is `_pdf(x).log()`, which underflows. Bind the native `ln_pdf` / `ln_pmf` whenever `statrs` has one, otherwise hand-write the body. |
| `sf` | **always**, override the default | Bind native `*CDF::sf` when present: better upper-tail accuracy than `1 - cdf`. |
| `log_cdf` / `log_sf` | **always**, override the default | `statrs` exposes neither, so there is nothing to bind and nothing safe to inherit; each is a hand-written Rust body. See [Contributing > Numerical stability](../contributing.md#numerical-stability). |
| `mean`, `variance`, `entropy` | Polars if closed-form | `n * p`, `loc`, `1 / rate`, `log(4 * pi * scale)`. Rust only where there is no closed form: a support sum, or log-gamma plus digamma. |
| `median` | override the default | The base default is `ppf(0.5)`. Bind native `Median::median` only where it agrees with scipy; Binomial's does not. |
| `std` | Polars, and override | The base default `variance().sqrt()` saturates long before the answer does. |
| `isf` | **always**, override the default | The base default `ppf(1 - q)` saturates long before the answer does, and it is value-keyed, so it carries the same arm-masking constraint as `ppf`. |

### Expose the conventional parameterisation, document the scipy mapping

A distribution takes the parameters it is conventionally defined by, which is also what `statrs` takes. A
scipy-spelled alias is never added: `Exponential` takes `rate`, not scipy's `scale = 1 / rate`, and gains nothing from
accepting both. An alias has to be coerced, validated, tested on both fast paths, and explained forever.

Where the two conventions genuinely differ, the divergence gets **exactly three mentions**: the class docstring, the
catalogue row in the [API reference](../reference/index.md#catalogue), and
[How-to / Migrate from scipy.stats](../how-to/migrate-from-scipy.md). Diverge from scipy's *convention* rather than
just its spelling only where scipy's is a footgun, and say why in the docstring.

### Sampling derives a fresh per-row RNG from `(root_seed, row_index)`

Every sampler needs one property: a deterministic, independent stream per row that depends only on
`(root_seed, row_index)`, never on position within a chunk. That makes output invariant to how Polars chunks or threads
the input, so `sample` is genuinely elementwise.

The generator is `Pcg64Mcg` (`rand_pcg`): cheap to construct (no key schedule), good statistical quality, and output
stable across releases and platforms, so seeded results stay reproducible. The root seed is resolved once per call
(`SysRng` when `seed=None`); each row then derives its own generator from `(root_seed, row_index)`.

This replaced an earlier "single `ChaCha20Rng` advanced once per row in iteration order" design, which coupled rows
across chunks (order-dependent, not streaming-safe). The naive fix, a `ChaCha20Rng` per row, made sampling markedly
slower (a key schedule plus a keystream block per draw). A one-shot hash-to-uniform would be cheaper still but only
serves distributions needing a single uniform per draw, so it is deliberately not the foundation.

### Constant parameters take a sampler fast path

`sample` ships a second plugin, `<name>_sample_scalar`, used when every parameter is a Python scalar. The general
sampler is built for the differentiator (column-valued parameters), but it makes the common constant-parameter case pay
for machinery it does not use: each scalar is broadcast to a full-length column, marshalled across FFI, and
re-validated on every row, and the distribution is rebuilt per row. For a cheap draw (uniform is one multiply-add) that
fixed overhead dominates the draw itself.

The fast path passes the constant parameters in `kwargs`, validates and builds the distribution once, and sends only the
row index as an input. It keeps the exact `(root_seed, row_index)` seeding and the same draw, so its output is
byte-identical to the per-row path for any seed; that equality is the contract, pinned by a property test
(`test_sample_scalar_fast_path_matches_per_row`) rather than left implicit. The result is less per-row work and lower
peak memory (no constant columns), with reproducibility and the chunk- and thread-invariance guarantees untouched.

It is the one deliberate exception to "parameters travel in `inputs`, not `kwargs`": admissible precisely because the
path is selected only when the parameters are known scalars, so nothing column-valued is ever forced into `kwargs`.

### Constant parameters validate once, not per row

The moments (`mean`, `variance`, `std`, `entropy`) and the value-keyed closed forms still assembled in Polars
(`Uniform`, `Geometric`) do not build a distribution; they compute a Polars expression. But they still route their
*validation* through a small Rust plugin (`normal_sigma`, `uniform_range`, `bernoulli_proba`,
`binomial_params`, `lognormal_sigma`, `exponential_rate`, `beta_params`) so an invalid parameterisation raises the same
`ComputeError` as the sampler and value-keyed methods rather than silently producing a nonsense moment (see "Invalid
parameters raise"). With column parameters that plugin runs over the parameter columns, validating each row.

For all-scalar parameters the same plugin is called on length-1 `pl.lit` inputs, so its elementwise closure runs once.
The validated quantity (or, for `Beta.entropy` and `Binomial.entropy`, the entropy itself) is returned behind a
`pl.when(...)` validity gate. Nothing in the expression is longer than one row, so the moment is a *scalar* column that
polars broadcasts wherever it meets a longer one; only the standalone height differs
(`df.select(Normal(0.0, 1.0).variance())` is one row). This needs no new plugin and no kwargs, only the existing
validators called on fewer rows. The two spellings can disagree in the last bit; see
[Numerical accuracy](accuracy.md#structural).

It is the same "constant parameters take a fast path" idea as the sampler, applied to validation: nothing leaves Rust,
and the raise contract is unchanged (pinned by `moment_test.py` and the `*_scalar` validation tests). For a constant,
"per row" and "once" are the same check.

### `samples` draws each row's array in one native call

`sample_iter` was rejected for the multi-draw loop body: it advances a single stream in row order, which couples rows
across chunks and breaks the invariance the per-row seeding exists to provide. `samples` instead uses one stream *per
row*: row `i`'s `size` draws are consecutive values from the `(root_seed, i)` stream, the same stream `sample` takes its
single draw from. That stays chunk-invariant because the stream is keyed by global row position; what remains rejected
is any stream shared across rows. The per-row stream is what makes `samples(size=1)` equal `sample` bit for bit and
`samples` prefix-stable in `size` (both pinned by property tests).

It also runs as a single native plugin call that fills the whole `Array(inner, size)` column in one pass, replacing an
earlier construction of `size` separate `sample` calls glued by `concat_arr`.

### Binomial sampling uses `rand_distr`, not `statrs`

`statrs` (still true on 0.19) implements `Distribution<u64> for Binomial` as `(0..n).fold(...)`, one uniform draw per
trial, so a sampled row costs `n` RNG draws. At `n = 10_000` that is 10,000 uniforms per row, turning the sampler
`O(n)`; the constant-factor wins over scipy held only for small `n`. The binomial sampler therefore draws from
`rand_distr::Binomial` (inversion for small `n*p`, BTPE otherwise, both `O(1)`-amortised), keeping sampling time flat
in `n`. This is the one place sampling
does not go through `statrs`; every value-keyed method (`pmf`, `cdf`, `ppf`, ...) still builds the `statrs` distribution.

### Invalid parameters raise, they never silently null

An invalid parameter value, scalar or one bad column row (`sigma <= 0`, `max <= min`, `p` outside `[0, 1]`, a
non-finite bound), maps the `statrs` constructor error through a `ComputeError` and fails the whole evaluation.

This reverses an earlier "produce null, keep the pipeline running" decision. Silently nulling hides a modelling error: a
user who does not check for nulls gets wrong answers downstream, and an invalid-parameter null is indistinguishable from
a legitimately-null input. Raising is loud, uniform across distributions, and uniform across scalar vs column inputs,
because scalars are coerced to columns and validated per row exactly like columns. Construction rejects only wrong
*types*. A closed-form distribution cannot raise from a bare `pl.Expr`, so it routes parameters through one small
validating plugin (see [Architecture / Plugin granularity](architecture.md#plugin-granularity)).

### Moments that are undefined return null; divergent ones return `+inf`

Every distribution shipped today has finite moments on its valid parameter range, so this policy does not bite yet; it
governs distributions on the roadmap. Three outcomes, and the distinction is what the quantity does rather than
whether the user asked a reasonable question:

* **Undefined**, permanently or only on part of the range (a Cauchy mean; a Student-t mean at `df <= 1`): **null**.
  It is Polars' own representation of "no value here", it is what scipy's `nan` maps to under the Polars idiom, and it
  keeps a parameter sweep across the threshold from dying on an exception.
* **Divergent** (a Pareto mean at `shape <= 1`): **`+inf`**, matching scipy. The integral has an answer and it is
  infinite, which is different from having none.
* **Invalid parameterisation**: raises, as everywhere else. This is the case the other two must not be confused with,
  so a moment with no formula at all still routes through the parameter validator: `Cauchy.mean()` is null for a valid
  `scale` and raises for a negative one.

**An undefined moment is null, not `NotImplementedError`.** An earlier version of this note said the opposite for the
permanently-undefined case. It was reversed because the two cases are indistinguishable to a user sweeping a
parameter, and because raising makes a mixed `select` unusable: one column having no mean should not fail the query.

The cost is real and belongs in the class docstring rather than being hidden: a null is silently absorbed by a
downstream `.mean()`, `.sum()` or `.drop_nulls()`, so a user who does not know `Cauchy.mean()` is null can lose rows
without noticing. `NotImplementedError` is reserved for a quantity this library has not implemented, such as a
`NegativeBinomial` entropy with no closed form at all.
