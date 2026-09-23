# Compute! Paris

## Title

Vectorized statistical distributions, built on the Polars engine

Proposed talk title on the slides: *Two giants and a seam: statistical distributions inside Polars, without becoming
a Rust expert*.

## Abstract

If you do statistical work in Polars today, you leave Polars to do it: `.to_numpy()`, a trip through `scipy.stats`,
and back. That is a reasonable thing to do. `scipy` broadcasts parameter arrays, so a different distribution per row
is already vectorised there. What you pay is where the result lands: the lazy query ends at that boundary, keeping
the result row-aligned through later joins and filters is your job, and an invalid parameter comes back as `NaN`
with no warning.

This talk walks through `polars-stats`, a Polars expression plugin that exposes `scipy.stats`-style distributions
natively inside Polars expressions, with column-valued parameters as a first-class feature. The maths is Rust
(`statrs`); the surface is Polars. The speaker is not a Rust expert, and the talk is built from that point of view:
what the two giants hand you for free (lazy evaluation, streaming, threads, `over`, and someone else's error
function), and the seam between them where every shipped bug actually lived.

Three seams, each a real release: row alignment that Polars does not do for a plugin, so a million-row result came
back with ten distinct answers and no error; a seeded sampler whose one shared stream reproduced the chunk layout
instead of the query; and a null-versus-raise contract that a Polars optimiser masked away until the closed forms
moved into Rust. Expect concrete code, the dates each mistake shipped and was fixed, a benchmark against `scipy`
read within one parameter regime with every disclaimer attached, and one rule to take back to your desk.

## Description

**The problem**: Polars has become a widely adopted dataframe engine for data engineering and ML feature pipelines,
but it has no native statistical distribution layer. Users fall back to three options:

1. round-trip through `scipy.stats` (ends the lazy query, materialises columns, hands alignment back to the caller),
2. Python UDFs via `map_elements` (slow, GIL-bound, not vectorised),
3. hand-rolled per-distribution expressions (duplicated and error-prone in the tails).

The first is what people actually do, and it deserves respect: `scipy` broadcasts parameter arrays, so the per-row
case (the probability of a reading under a Normal whose mean and standard deviation are stored as columns) is already
vectorised there. The result comes back as a NumPy array. The `collect()` cuts the query in half, pushdown stops at
that boundary, keeping the result row-aligned through joins, filters and `over` is the caller's job, and NumPy has no
null, so an invalid parameter returns `NaN` instead of raising.

**What we built** is a Polars expression plugin that puts `scipy.stats`-style distributions directly into Polars
expressions. Two properties set it apart from the sampling-focused plugins that already exist:

* Column-valued parameters across the whole surface: any parameter can be a scalar or a Polars expression, for
  `pdf` / `cdf` / `sf` / `ppf` / `isf`, their log variants, the moments and sampling alike.
  `Normal(mu=pl.col("mu"), sigma=pl.col("sigma")).sf(pl.col("x"))` evaluates one distribution per row, fully
  vectorised.
* Lazy-native: every method returns a `pl.Expr`, so nothing materialises and both the in-memory and streaming engines
  are supported.

**Point of view**: the speaker is not a Rust expert, and a good part of the Rust layer was written with AI assistance,
as the README says. The talk is about what that makes possible and what it makes dangerous. Two giants do the heavy
lifting: Polars provides the engine (lazy evaluation, streaming, the thread pool, `over` and `group_by`) and `statrs`
provides the maths (`pdf`, `cdf`, `inverse_cdf`, `ln_pdf`, sampling). The Rust file for a distribution is short and
its maths is a handful of one-line calls. The longest files in the repository contain no maths at all: they align
inputs, gate dtypes and decide how a row gets its seed. That is where the talk spends its time.

**What the audience will take away**:

1. Plugin anatomy: how a `pyo3-polars` expression plugin is wired across three layers (Python API, Rust FFI, the
   `statrs` math layer), what each layer gives you without writing it, and why parameters must travel as
   length-matched inputs rather than static kwargs to stay column-valued.
2. Seam 1, row alignment: Polars broadcasts nothing into a plugin and the elementwise helpers truncate to the shortest
   input. Python-side scalar padding fixed `sigma=0.5` and missed `pl.lit(0.5)` and `.first()`, so a million-row
   frame came back with ten distinct answers and no error. Alignment moved into Rust, by length, and an all-constant
   expression became a one-row scalar column, which was a breaking release.
3. Seam 2, reproducible sampling: a single `ChaCha20` stream advanced in row order reproduced the chunk layout, not the
   query, so the same seed gave different columns on the two engines. Sampling is now keyed on `(seed, row index)`
   with a cheap per-row `Pcg64Mcg`, the row index travels as an ordinary input, and a seeded column repeats across
   runs, chunk layouts, thread counts, operating systems and both engines.
4. Seam 3, the null and error contract: "null the bad row and keep going" makes an invalid parameter indistinguishable
   from missing data, so it was reversed to "null in, null out; invalid parameter raises". A `pl.Expr` cannot raise
   per row, the validating plugin sat inside a `when` arm, and polars 1.44 optimised arms so they never see the rows
   they do not select. The library capped `polars<1.44`, moved every value-keyed closed form into Rust, and lifted the
   cap.
5. Honest costs: a benchmark against the classic frozen `scipy` API on one machine, read within one parameter regime,
   with the cells `scipy` wins shown alongside the ones it does not; the `statrs` limits the library inherits and
   files upstream rather than patching locally, found by an `mpmath` oracle at 50 digits; and the catalogue, still a
   short list against `scipy`'s hundred-plus.

Each seam is a small, transferable lesson about building a numerical library on top of someone else's maths and
someone else's engine: borrow both, and write down what neither promises you before it becomes a bug.
