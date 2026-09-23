---
theme: seriph
title: Fifty-five lines of caveats
info: |
  Every bug polars-stats shipped was already in its first commit, written down as handled or "not supported".
  compute! Paris 2026 · Francesco Bruzzesi
colorSchema: light
transition: fade
mdc: true
themeConfig:
  primary: '#211C4F'
layout: cover
background: '#211C4F'
---

# Fifty-five lines of caveats

Every bug `polars-stats` shipped was already in its first commit

<div class="mt-12 muted">

Francesco Bruzzesi · compute! Paris 2026 · `pip install polars-stats`

</div>

<!--
Title slide. Do not introduce yourself here. Go straight into the scene on the next slide.
-->

---

# 26 April 2026, 23:51. One method, 106 lines, tests pass. Find the three bugs.

```python {all|11-14|18-19|5-7}
class Bernoulli:
    def __init__(self, p: float | pl.Expr) -> None:
        if isinstance(p, pl.Expr):
            self._p = p
        elif isinstance(p, float):
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"p must be in the [0, 1] range, found {p}")
            # Materialise scalar p to one row per output row.
            # `is_elementwise=True` does NOT broadcast inputs before the plugin runs
            # (it only broadcasts the *result*), so a length-1 `pl.lit(p)` would make
            # the plugin draw a single sample which polars then repeats across the whole frame.
            self._p = pl.repeat(p, n=pl.len())

    def sample(self, seed: int | None = None) -> pl.Expr:
        """Draw one Bernoulli sample per row.

        Reproducibility under ``seed`` assumes a single-chunk input series;
        chunked / streaming inputs are not supported.
        """
        return register_plugin_function(args=[self._p], function_name="bernoulli_sample", kwargs={"seed": seed}, ...)
```

<div class="muted text-sm mt-1">Condensed from commit <code>0e66820</code>. The Rust side: 55 lines, one <code>ChaCha20Rng::seed_from_u64(seed)</code>, one <code>for i in 0..n</code>, and a second validator for columns.</div>

<!--
Sunday, 26 April, eleven fifty-one at night. I push the first commit of a library called polars-stats. It does one thing: draw a Bernoulli sample per row of a Polars frame, with a probability that can be a column. Fifty-five lines of Rust. Fifty-one of Python. The tests pass.

Look at the Python for a moment, because I am going to ask you to find the bugs. There are three.

[click] Here is the first. A comment. It says, correctly, that Polars does not broadcast inputs into a plugin, so a length-one pl.lit(p) would make the plugin draw once and Polars would repeat that one draw across the frame. So the code pads every Python float with pl.repeat. Handled.

[click] Second. The docstring. "Reproducibility under seed assumes a single-chunk input series; chunked / streaming inputs are not supported." A known limitation, written down.

[click] Third. A Python float is checked in Python, eagerly, with a ValueError. A column is checked in Rust, per row, with a different error. Two validators, two messages, and nobody has decided which one is the contract.

Everything that went wrong in this project is on this slide. The first one shipped as a bug in a published release, with a comment above it that describes the exact mechanism. The second one meant the same seed gave different numbers on the two engines. The third one turned into the longest argument in the repository and a Polars release that switched my validation off from the outside.
-->

---
layout: statement
---

# I am not a Rust expert.

A good part of the Rust in this library was written with AI assistance. It says so in the README.
What I vouch for is behaviour, pinned by tests, and this talk is how each of those tests came to exist.

By the end you will be able to look at your own "known limitation" comments and tell which are <span class="ox">bugs with a delivery date</span>.

<!--
One sentence about me, because it makes the rest credible: I am not a Rust expert, and a good part of the Rust in this library was written with AI assistance. It says so in the README. What I vouch for is behaviour, pinned by tests, and this talk is the story of how each of those tests came to exist. By the end you will be able to look at your own "known limitation" comments and tell which of them are bugs with a delivery date.
-->

---

# "Just use scipy" is right. The cost is where the result lands, not speed.

```python
mu, sigma, x = (df[c].to_numpy() for c in ("mu", "sigma", "value"))
tail = stats.norm(loc=mu, scale=sigma).sf(x)  # one distribution per row, vectorised, in C
df = df.with_columns(anomaly=pl.Series(tail))
```

<v-clicks>

- **The plan ends there.** `collect()` before scoring; the optimiser never sees the second half.
- **Alignment is yours.** Through a join, a filter, an `over`, keeping the result row-aligned is your job.
- **NumPy has no null.** Show of hands: `stats.norm(loc=0, scale=-1).sf(1.0)` returns?

</v-clicks>

<v-click>

```python
>>> stats.norm(loc=0.0, scale=-1.0).sf(1.0)
nan                                            # scipy 1.18.1, warnings on, none emitted
```

</v-click>

<!--
Why write this at all? Before the library I did what you do. Pull the columns to NumPy, call scipy.stats, wrap the result back into a Series. And that is a good idea. scipy broadcasts. If mu and sigma are arrays, norm(loc=mu, scale=sigma).sf(x) scores every element against its own distribution, in C, no loop. Per-row parameters were solved twenty years ago.

What is wrong with it is not speed.

[click] The plan ends there: you collect() before scoring, and the optimiser never sees the second half.

[click] Alignment is yours: once the values leave the frame, keeping the result row-aligned through a join or an over is your job.

[click] And NumPy has no null. Show of hands: norm(loc=0, scale=-1).sf(1.0) returns?

[click] nan. No warning. A negative standard deviation is a modelling error, and it travels through your pipeline dressed as missing data. Hold that thought.
-->

---

# Keep it a `pl.Expr` and the scoring is one node in the plan. The filter moves below it.

```python
lf = (
    pl.scan_parquet("readings.parquet")
    .with_columns(anomaly=ps.Normal("mu", "sigma").sf("value"))
    .filter(pl.col("device") == "sat-07")
    .filter(pl.col("anomaly") < 1e-3)
)
```

```text {all|4|6-7}
FILTER col("anomaly") < 0.001
FROM
   WITH_COLUMNS:
   [col("value")..../_internal.abi3.so:normal_sf([col("mu"), col("sigma")]).alias("anomaly")]
    Parquet SCAN [readings.parquet]
    PROJECT */4 COLUMNS
    SELECTION: col("device") == "sat-07"
```

<v-click>

```python
lf.collect(engine="in-memory").equals(lf.collect(engine="streaming"))  # True, 911 rows each
lf.sink_parquet("scored.parquet")  # streams file to file, never fits in memory
```

</v-click>

<!--
So the idea was to keep the result a pl.Expr, and I want to show you exactly what that buys, because "lazy and streaming" is a slogan until you look at a plan. Here is a Parquet file of telemetry. Scan it. Add an anomaly score, the upper-tail probability under each row's own Normal. Filter to one device. Filter to scores below one in a thousand. Ask Polars to explain.

[click] The scoring is one node. normal_sf, a function in a shared library, with three column inputs.

[click] Look below it. The device filter I wrote after the scoring has moved into the scan, as a selection on the Parquet reader, so only that device's row groups are decoded. The score filter stays above, because it depends on the score. Nobody materialised anything.

[click] Now collect it twice, once on the in-memory engine, once on the streaming engine. Same frame, row for row. Replace collect() with sink_parquet() and the whole thing streams from one file to another, in morsels, and the frame never has to fit in memory. Every method in the library returns an expression, so this works for a sample, a quantile, a log-density, anything. That is the whole product, and I did not write any of it.
-->

---
layout: two-cols-header
---

# Two giants hand you all of that. Polars the engine, statrs the maths.

::left::

### Polars

A plugin is a Rust function with one attribute.
It takes `&[Series]`, it returns a `Series`.

For signing that contract you get, without writing any of it:

- lazy evaluation and the optimiser
- the streaming engine
- the thread pool
- `over` and `group_by`, one call per partition

::right::

### statrs

The Rust crate that does what `scipy.stats` does.

```rust
let dist = Normal::new(mu, sigma)?;
dist.sf(x);
dist.cdf(x);
dist.inverse_cdf(q);
dist.ln_pdf(x);
dist.sample(&mut rng);
```

Someone else already got the error function right.

<!--
Here is what I did write, and what I did not. A Polars plugin is a Rust function with one attribute. It receives Series, it returns a Series. For signing that contract you get lazy evaluation, the optimiser, the streaming engine, the thread pool, and over and group_by, because Polars calls your function once per partition. That is the first giant.

The second is statrs, the Rust crate that does what scipy.stats does: Normal::new(mu, sigma), then cdf, sf, inverse_cdf, ln_pdf. Someone else got the error function right.
-->

---

# The longest files in the repository contain no maths. All three caveats live in them.

<div class="grid grid-cols-2 gap-10 mt-6">
<div>

**One maths file per distribution** · `normal.rs`, `beta.rs`, `weibull.rs`, ...

<div class="big">~300</div>
lines each. The maths in each is a dozen one-line calls into `statrs`.

</div>
<div>

**Two boundary files** · `mod.rs` + `rng.rs`

<div class="big">965</div>
lines. Align inputs, gate dtypes, decide how a row gets its seed.
Zero distribution maths. The only code that is mine.

</div>
</div>

<div class="muted mt-8 text-sm">Line counts at v0.1.0. They drift; the shape does not.</div>

<!--
Look at the tree today. One Rust file per distribution, about three hundred lines, and the maths in it is a dozen one-line calls into statrs. The two longest files in the repository, close to a thousand lines between them, contain no maths at all. One aligns inputs and checks dtypes. The other decides how a row gets its seed. That is the boundary between the two giants, it is the only code that is mine, and all three caveats from the first commit live in it. Let us pull the first thread.
-->

---
layout: section
---

# Caveat 1

## "Reproducibility under seed assumes a single-chunk input series; chunked / streaming inputs are not supported."

<!--
The docstring said chunked input was not supported. What can that sentence mean to a caller?
-->

---

# Same seed, same frame, two engines. Same numbers?

````md magic-move
```rust
// day one: one stream, seeded once, advanced once per row in iteration order
let mut rng = ChaCha20Rng::seed_from_u64(seed);
for p in rows {
    out.push(Bernoulli::new(p)?.sample(&mut rng));
}
```

```rust
// what polars actually does: calls the plugin once per chunk, in thread-pool order
for chunk in chunks_in_whatever_order() {          // the streaming engine cuts into morsels, differently
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    for p in chunk {
        out.push(Bernoulli::new(p)?.sample(&mut rng));
    }
}
```
````

<v-click>

There is no argument a caller can pass to make the input single-chunk. "Not supported" meant: **the seed reproduces the chunk layout, not the query.**

</v-click>

<!--
You write sample(seed=42). Polars decides how many chunks your frame is in. The streaming engine cuts it into morsels, and it cuts differently from the in-memory engine. The thread pool decides which chunk the plugin sees first. There is no argument you can pass to make the input single-chunk. So "not supported" meant: the seed reproduces the chunk layout, not the query. Prediction. Same seed, same frame, collect() twice, once per engine. Same numbers? Hands up.

[click] No. Row zero of chunk three got the fourth draw on one engine and the four-thousandth on the other. And to be fair to that first design, it is what anyone writes: one generator, advanced per row. It is correct for a list. It is wrong for an engine that owns the iteration order, and the streaming engine owns it more than most.
-->

---

# One stream per row, keyed on (seed, row index). The fix cost 10 to 20x, then nothing.

```python
ROW_INDEX_EXPR = pl.int_range(0, pl.len(), dtype=pl.UInt64)  # Python adds it as a plugin input
```

```rust {all|3}
for (p, index) in rows {
    // first: ChaCha20 per row. a week later: Pcg64Mcg per row, a handful of integer ops to construct.
    let mut rng = row_rng(seed, index);
    out.push(Bernoulli::new(p)?.sample(&mut rng));
}
```

<v-clicks>

- Commit #7, first line: *"Replace per-row ChaCha20 construction (**10-20x regression**) with a shared Pcg64Mcg helper."* Seeded values changed. The contract did not.
- **The index is the one input that must never broadcast.** The streaming engine hands the plugin one-row morsels, so a length-1 index makes every row "row zero": one draw at full height, no error. Sized by the call, in Python, before anything crosses.
- **CI runs the whole suite twice, once per engine.** They chunk a plugin's inputs differently; a chunk-boundary bug passes on one and fails on the other.
- Today: 1 chunk or 10, in-memory or streaming, macOS or Linux, same column. `samples(size=1)` equals `sample` bit for bit. The docstring sentence became a property test.

</v-clicks>

<!--
The fix, a month in, is to stop having one stream. Every row derives its own generator from the seed and its global position. Where does the position come from? Polars does not tell a plugin which rows it is holding. So the Python side adds one input column, an integer range, and passes it in like any other parameter. The plugin never asks where it is. It reads it.

[click] Then the bill. A ChaCha20 generator per row means a key schedule per draw,

[click] and the commit that fixed it a week later says "10-20x regression" in its first line. Rust has cheaper generators. Pcg64Mcg is a handful of integer operations to construct, it passes the usual statistical batteries, and its output is stable across releases and platforms. With a mixing step on (seed, index), the per-row version costs what the one-stream version did. Seeded values changed; the contract did not.

[click] Two things about the streaming engine came out of this thread, and both are now tests. First, the row index is the one input that must never be broadcast. If it arrives as length one, every row is row zero, and you get one draw repeated at full height with no error. The plugin cannot repair that, because the streaming engine splits the call into one-row morsels and a flattened index is all it ever sees. So the index is sized by the call, in Python, before anything crosses the boundary.

[click] Second, since August the test suite runs twice in CI, once per engine, because they chunk a plugin's inputs differently and a bug at a chunk boundary passes on one and fails on the other.

[click] Today: one chunk or ten, in-memory or streaming, macOS or Linux, same column. samples(size=1) equals sample bit for bit, because it is the same stream. That docstring sentence is gone. It became a property test.
-->

---
layout: section
---

# Caveat 2

## "A length-1 `pl.lit(p)` would make the plugin draw a single sample which polars then repeats across the whole frame."

<!--
Second thread, the comment. Version 0.0.1 goes to PyPI in August. I shipped it early, mostly to claim the name.
-->

---

# A million rows scored. Count the distinct answers.

```python {1-6|8}
readings: pl.LazyFrame  # 1_000_000 rows: value, mu, sigma
score = ps.Normal(mu="mu", sigma=pl.lit(0.5)).sf("value")

scored = readings.with_columns(anomaly=score).collect()
scored.height
# 1000000

scored["anomaly"].n_unique()
```

<v-click>

<div class="big mt-6">10</div>

</v-click>

<v-click>

<div class="mt-2">With four threads instead of ten: <span class="ox">12</span>. No error. No warning. Float64, right length.</div>

</v-click>

<!--
Here is the telemetry frame again, a million rows, every row with its own baseline. One line scores it. A million rows in, a million out, Float64, right length.

Count the distinct values in that column.

[click] Ten.

[click] Run it with four threads instead of ten. Twelve.
-->

---

# The comment described the mechanism. `pl.repeat` covered one of four spellings.

| parameter | length received by the plugin | covered by `pl.repeat`? |
|---|---|---|
| `sigma=0.5` | 1 000 000 | yes |
| `sigma=pl.lit(0.5)` | <span class="ox">1</span> | no |
| `sigma=pl.col("sigma").first()` | <span class="ox">1</span> | no, and Python cannot know until runtime |
| `sigma=pl.col("sigma").max()` | <span class="ox">1</span> | no |

<v-click>

`try_ternary_elementwise` is a zip. Output length **1**, correctly computed. Then `with_columns` broadcast the one-row result, once per chunk. Ten chunks, ten answers.

</v-click>

<v-click>

**I had tested it.** On a small frame, on the streaming engine, which hands the plugin one row at a time. One output row per one-row morsel adds up to exactly the frame height. Green. A small frame on the streaming engine is not evidence about lengths.

</v-click>

<!--
Go back to the comment. "A length-one pl.lit(p) would make the plugin draw once and Polars would repeat it across the frame." I wrote that on day one. The code padded every Python float with pl.repeat, and a Python float was fine. What I had not thought through is that a float is not the only way to hand me a length-one input. pl.lit(0.5) is length one. pl.col("sigma").first() is length one. .max() is length one. And Python cannot know that when it builds the plan, because the length of .first() is a runtime fact.

[click] So the plugin received a million values, a million means and one sigma. The helper that walks them together is a zip, and a zip stops at the shortest input. It returned one row, correctly computed, and with_columns did what it does with a one-row result: broadcast it. Once per chunk. Ten chunks, ten answers.

[click] And here is the part I find most uncomfortable. I had tested this. On a small frame. On the streaming engine, which hands the plugin one row at a time, so one output row per one-row morsel adds up to exactly the frame height. The test passed. A small frame on the streaming engine is not evidence of anything about lengths.
-->

---

# Alignment moved to where the lengths are known: Rust, line one. It was a breaking release.

````md magic-move
```python
# 0.0.1, Python, plan time: pad every Python scalar
def coerce_param(value, *, name):
    if isinstance(value, float):
        return pl.repeat(value, n=pl.len())   # sigma=0.5: fine
    if isinstance(value, pl.Expr):
        return value                          # sigma=pl.lit(0.5): length 1, nobody notices
```

```rust
// 0.0.2, Rust, runtime: align by length, whatever the expression was
pub(crate) fn align_inputs(inputs: &[Series]) -> PolarsResult<Cow<'_, [Series]>> {
    let anchor = inputs.iter().find(|s| s.len() != 1);              // the call's row count
    for s in inputs {
        polars_ensure!(s.len() == 1 || s.len() == anchor.len(), ShapeMismatch: "...");
    }
    Ok(inputs.iter().map(|s| if s.len() == 1 { s.new_from_index(0, anchor.len()) } else { s.clone() }).collect())
}
```
````

<v-click>

```python
df.select(ps.Normal(0.0, 1.0).mean())
# shape: (1, 1)      an all-constant expression is one row now, as any polars constant is
```

I had handled the caveat where I could see it, in Python at plan time. It was only true in Rust at runtime.

</v-click>

<!--
The fix is small and it is in Rust, because Rust is where the lengths are known. align_inputs runs before any cast. Every length-one input is broadcast up to the row count of the call; two inputs of different lengths that are not one raise a shape error instead of zipping. It aligns by length, not by what the expression looked like in Python.

[click] It was a breaking release. If every input is constant, Normal(0.0, 1.0).mean() in a select is now one row, which is what Polars does with a constant. The padding had made it a full column. The new behaviour is right and it was still a change.

I had handled the caveat where I could see it, in Python, at plan time. It was only true in Rust, at runtime.
-->

---
layout: section
---

# Caveat 3

## Two validators, two messages, and no decision about which one is the contract.

<!--
Third thread: two validators, one in Python for floats and one in Rust for columns, and no decision about which one was the contract. The decision is about what should happen when sigma is negative on row four hundred.
-->

---

# "Null the bad row and keep going" rebuilds the scipy `nan` inside Polars.

<div class="grid grid-cols-2 gap-8 mt-4">
<div>

**First contract** (friendly)

| input | result |
|---|---|
| `sigma = null` | null |
| `sigma = -1.0` | null |

</div>
<div v-click>

**Current contract**

| input | result |
|---|---|
| `sigma = null` | null, always |
| `sigma = -1.0` | `ComputeError`, whole evaluation |

</div>
</div>

<v-click>

Missing data and a broken estimator upstream must not be the same row. One check, in Rust, shared by floats and columns.

</v-click>

<!--
My first answer was the friendly one. Null the row. Keep the nightly job running. Then I looked at the result and realised I had rebuilt the scipy problem from earlier: a row that is null because sigma was missing and a row that is null because sigma was negative are the same row. One is missing data. The other is a bug upstream.

[click] So the contract flipped. Null in, null out, always. Invalid parameter, raise, fail the whole evaluation, name the parameter and the value.

[click] Same rule for a float and for a column, because both go through one check, in Rust.
-->

---

# A `pl.Expr` cannot raise per row. The validator went inside an arm. Then Polars 1.44.

```python {all|4|4}
# Uniform.cdf, assembled in Python: the only Rust is the validator
pl.when(x < min).then(0.0)
  .when(x >= max).then(1.0)
  .otherwise((x - min) / uniform_range(min, max))     # Rust: raises on max <= min
```

<v-click at="2">

Polars 1.44: an arm is **masked** on the rows it does not select, and **skipped** if no row selects it.
Row with `x < min` and `max <= min`: the validator receives nothing, and `cdf` returns `0.0`.
**23** (distribution, method) pairs, both engines. Exactly the distributions assembled in Python, and only those.

</v-click>

<v-click>

1. Cap `polars<1.44`, ship. A sweep test: 384 assertions pass on 1.43.2, **106 fail** on 1.44.1. No xfail.
2. Port those distributions' value-keyed closed forms to Rust. Lift the cap. Two weeks. Done.

</v-click>

<!--
Then the mechanism, because a decision you cannot enforce is a comment. A pl.Expr cannot raise per row. So for the distributions whose closed forms were Polars arithmetic, I routed the parameters through a tiny Rust plugin whose only job was to raise,

[click] and I put it where it seemed to belong: inside the when / then / otherwise that assembled the formula.

[click] Polars 1.44 shipped an optimisation, and a good one. A when arm no longer sees rows it does not select, and an arm nobody selects is not evaluated at all. My validator was in an arm. On a row where the condition sent the value elsewhere, it received nothing, said nothing, and the method returned a number for a negative parameter. Twenty-three distribution and method pairs, on both engines. Exactly the distributions whose closed forms lived in Python, and only those. The ones that had always been Rust lost nothing. The sweep that found it passes 384 assertions on 1.43 and fails 106 on 1.44.

[click] Two moves. Cap Polars below 1.44 and ship, with a test that goes red the day the cap is lifted. Then move every value-keyed closed form into Rust, where the check runs on the parameter column before any row is built, and lift the cap. Two weeks. Done.
-->

---

# Except it was not. One more `when` sat above every hook, Rust or not.

```python {all|1-3}
pl.when(value.is_null()).then(None)          # the null / NaN contract, applied uniformly
  .when(value.is_nan()).then(float("nan"))   # ... above EVERY public method
  .otherwise(self._cdf(value))               # the plugin, ported to Rust or Rust from day one
```

<v-clicks>

- On a null evaluation point the plugin **never ran**, so a negative `sigma` on that row was never seen. For every distribution. Including the ones that had been Rust since the start.
- Moving the maths into Rust changed nothing about a wrapper that sat above the maths.
- The wrapper is deleted. The plugin answers the null and `NaN` rows itself, before it computes anything.

</v-clicks>

<v-click>

```text
sigma = [1.0, None, -1.0]
ComputeError: sigma must be finite and strictly positive, got -1        # and the None row stays null
```

**A contract you cannot enforce from inside the engine is a comment.** It may hold for a while. It has a delivery date.

</v-click>

<!--
Except it was not. Above every public method, ported or not, sat one more Python expression:

[click] when the value is null, then null; when it is NaN, then NaN; otherwise, the plugin. Friendly, uniform, and an arm.

[click] On a null evaluation point the plugin never ran, so a negative sigma on that row was never seen, for every distribution, including the ones that had been Rust from the start.

[click] Moving the maths into Rust changed nothing about a wrapper that sat above the maths.

[click] The wrapper went. The plugin answers the null and NaN rows itself, before it computes anything.

[click] That is the rule from this thread, and it is the docstring from April coming back: a contract you cannot enforce from inside the engine is a comment. It may hold for a while. It has a delivery date.
-->

---
layout: section
---

# What was not written down

<!--
Those three were written down. Two more things were not, because I did not know to write them.
-->

---

# The 50-digit audit found 11 defects where 3 were budgeted. The algebra was the worst offender.

<div class="grid grid-cols-2 gap-10 mt-4">
<div>

**Budgeted** <div class="big">3</div>

</div>
<div>

**Found** <div class="big">11</div>

</div>
</div>

<v-clicks>

- The special functions (`statrs`) came back clean. The elementary closed forms did not: nobody had checked `1 - (1 - p)` for `p = 1e-12`. It is exactly `0.0` in `float64`.
- Two findings were not inaccuracy. `Beta.ppf` far in the lower tail **panicked**, and a panic inside a plugin aborts the query, not the row. `Binomial.entropy` returned **`NaN`** on valid parameters.
- Nine of the eleven were one- to three-line fixes. A curated grid of test points had reached none of them.

</v-clicks>

<!--
My confidence was in the wrong place. The maths I was nervous about was the special-function maths: the error function, the incomplete beta. That was statrs, and I had read enough of it to trust it. The maths I was relaxed about was the algebra. Uniform.cdf is (x - a) / (b - a). Geometric.sf is (1 - p) ** k. You can check that on paper.

In August I stopped checking on paper and swept every method against mpmath at fifty digits, at inputs many decades past where scipy itself saturates. I had budgeted for three defects. It found eleven.

[click] The special functions came back fine. The elementary closed forms were the worst offenders by count. Nobody had thought to check 1 - (1 - p) for a p of ten to the minus twelve. It is exactly zero in float64. The tail was right on paper and gone in the machine.

[click] And two of the eleven were not inaccuracy at all. Beta.ppf far in the lower tail panicked, and a panic inside a plugin does not fail a row, it aborts the query. Binomial.entropy returned NaN on valid parameters.

[click] A curated grid of test points had never reached either. An argument that a method is safe is a hypothesis, and the sweep is the experiment.
-->

---

# An unsupported dtype did not raise. It aborted the interpreter.

```python
df.with_columns(ps.Normal("mu", "sigma").pdf(pl.col("value").cast(pl.Decimal(10, 2))))
# (process exits)                          no traceback, no test failure, 95% coverage
```

<v-clicks>

- A dead process reports nothing. Coverage is not evidence of safety at an FFI boundary.
- Fix: a Cargo feature that makes an unsupported dtype raise `InvalidOperationError`. Cost: **+3.4 MiB** of binary.
- Cheap, for the difference between an exception and a disappearing process.

</v-clicks>

<!--
The second thing nobody wrote down. In September a Decimal or Categorical value column did not raise. It aborted the interpreter. No traceback, no test failure, because a dead process reports nothing.

[click] Coverage was at 95 percent, and it was not evidence of anything at that boundary.

[click] The fix is a Cargo feature that makes an unsupported dtype raise, and it costs three and a half megabytes of binary.

[click] Cheap, for the difference between an exception and a disappearing process.
-->

---

# The numbers, with the disclaimers printed first.

<div class="disclaimer mb-4">

One laptop (Apple M5, 10 cores, 24 GiB), run 2026-09-22. polars 1.44.2, scipy 1.18.1, numpy 2.5.3, polars-stats 0.1.0 release build.
1 000 000 rows. **Wall time only**, median of 50 runs, no memory figures. scipy side is the classic frozen API (`norm(loc, scale).sf(x)`), which is not scipy at its fastest: the newer random-variable API measured earlier at roughly 1.3 to 1.9x quicker on these methods. polars-stats side includes building the query and `collect()`. Read a cell only against the cell beside it.

</div>

| regime | method | polars-stats p50 | scipy frozen p50 | ratio |
|---|---|---|---|---|
| column | `sf` | 2.9 ms | 13.2 ms | 4.5 |
| column | `log_sf` | 3.7 ms | 19.6 ms | 5.3 |
| column | `sample` | 1.7 ms | 4.3 ms | 2.6 |
| column | `mean` | 0.8 ms | 2.6 ms | 3.1 |
| scalar | `mean` | 0.11 ms | 0.01 ms | <span class="ox">0.08</span> |

<!--
The numbers, disclaimers first because they matter more.

One laptop, an Apple M5, run this week. Polars 1.44.2, scipy 1.18.1, release build. One million rows. Wall time only, median of fifty, no memory figures. The scipy side is the classic frozen API, which is what most code looks like and is not scipy at its fastest; the newer random-variable API measured earlier at roughly 1.3 to 1.9x quicker on these methods. The Polars side includes planning and collect(). Read a cell only against the cell beside it.

Within those limits: sf on a million rows with column parameters, about 3 milliseconds against about 13. Sampling, about 2.5x. And the cell I like most: a constant-parameter mean(). scipy wins by 10x, because it returns the parameter you gave it and a Polars query costs a tenth of a millisecond to plan. If you call a moment on constants a million times in a loop, scipy is the right tool and this library is the wrong one.
-->

---

# Three bills nobody puts on the benchmark slide.

<v-clicks>

- **The fix made one regime slower.** A `pl.lit` parameter was fast because polars constant-folded it, and that folding is what hid the validator. After the port it costs **2.5 to 3.6x** a Python float (10M rows, `DiscreteUniform.pmf`: 2.2 / 7.0 / 8.1 ms scalar / column / broadcast). The speed and the bug had one root. I kept the fix and wrote the number down.
- **The audit count went up as the library got better.** July headline: 34,000 probes, 0 non-`OK`. Mid-September: **~1,057 non-`OK` of 34,757**. Every row maps to a documented caveat with a regime and a magnitude. That mapping is the bar, not zero. Do not quote the July number.
- **You inherit the giant's limits, and its fixes.** `Beta.ppf` below `q ~ 1e-165` panics; in `[1e-150, 1e-60]` at some shapes it does not return (15 s for one row). Filed as statrs#435, documented, **not patched locally**. The other way: `statrs` 0.19.1 fixed three defects we reported, 17 days after 0.0.1. When #435 lands, the bullet comes out.

</v-clicks>

<!--
Now the bills nobody puts on the benchmark slide.

[click] The port that fixed the validator made one regime slower. A pl.lit parameter had been fast because Polars constant-folded it through pure arithmetic, and that folding is exactly what stopped the validator from running. After the port, pl.lit takes the per-row path and costs two and a half to three and a half times a Python float. The speed and the bug had one root. Buying the speed back meant keeping the bug. I kept the fix and wrote the number down.

[click] The audit. In July the headline was "34,000 probes, zero non-OK". In mid-September it was about 1,057 non-OK rows out of 34,757. The library got better in between. The instrument got better faster: more distributions, wider sweeps, inputs decades further into the tails. Every one of those rows maps to a documented caveat with a regime and a magnitude, and that mapping is the bar, not zero. Do not quote the July number. I am telling you so you do not.

[click] Last bill: what you inherit from a giant. Beta.ppf below ten to the minus one-sixty-five panics; between ten to the minus one-fifty and ten to the minus sixty at some shapes it does not return, fifteen seconds for a single row. That is filed upstream as statrs 435, it is in the docs with the regime and the magnitude, and it is not patched here. The deal runs both ways: statrs 0.19.1 fixed three defects we had reported, seventeen days after 0.0.1. When 435 lands, the bullet comes out.
-->

---

# What is left is not more of the same. And nobody is using it yet.

<div class="grid grid-cols-2 gap-8 mt-2">
<div>

**Shipped so far needed** an error function and an incomplete beta. `statrs` had both.

**What remains needs** special functions the library does not bind yet:

- a regularized incomplete gamma, for one family and the two that delegate to it
- a **log-space** incomplete beta, for two more families, and it fixes the last two bad tails already shipped (`Beta` / `Binomial` `log_cdf`). Written. Parked, waiting on upstream.

The catalogue itself is in the reference docs. It will have changed by the time you watch this.

</div>
<div>

**Three loops with no end**

- every `statrs` or Polars release: bump, re-audit, delete the workarounds it made redundant
- every new distribution: the 50-digit sweep before merge, not after
- every performance change: A/B against an untouched control, on a quiet machine

<div class="mt-6 disclaimer">

I do not know of a single production user. Every priority here is inferred from the code. The first real workload will reorder something.

</div>

</div>
</div>

<!--
Where it stands. Version 0.1.0 is out. The catalogue is in the reference docs, and I am not going to read it to you, because it will have changed by the time this recording is online.

Where it goes. What is left is not more of the same. Everything shipped so far needed, at most, an error function or an incomplete beta, and statrs had both. What remains needs special functions the library does not bind yet: an incomplete gamma for one family, and a log-space incomplete beta for another. That second one also fixes the last two bad tails in what is already shipped, the log_cdf of Beta and Binomial. It is written. It is parked, because the decision to report upstream rather than patch locally means waiting for statrs to say where it should live.

Then three loops with no end state. Every statrs or Polars release gets a bump and a re-audit, and every workaround an upstream fix makes redundant gets deleted. Every new distribution gets the fifty-digit sweep before it merges, not after. And every performance change gets an A/B against an untouched control on a quiet machine, because I have already been fooled once by a block order that faked a twenty percent regression.

And the honest line at the bottom of the plan: I do not know of a single production user. Every priority I just gave you is inferred from the code. The first real workload will reorder something, and I would like it to be yours.
-->

---

# 26 April, revisited. Not one of the three is a comment anymore.

| day one, 106 lines | where it lives at 0.1.0 |
|---|---|
| `# a length-1 pl.lit(p) would make the plugin draw once ...` `pl.repeat(p, n=pl.len())` | `align_inputs`, the first line of every plugin, and a broadcast test that sweeps `pl.lit`, `.first()` and `.max()` on both engines |
| `"""... chunked / streaming inputs are not supported."""` | one `Pcg64Mcg` per `(seed, row_index)`, and a property test across chunk layouts and both engines |
| `raise ValueError(...)` in Python, a different error in Rust | one check, in Rust, that owns every row, and a sweep that went red the day Polars changed |

<!--
Back to the twenty-sixth of April. Fifty-five lines of Rust, fifty-one of Python, three caveats. Here is where each one lives today. The padding comment became align_inputs, the first line of every plugin, and a broadcast test that sweeps pl.lit, .first() and .max() on both engines. The "not supported" docstring became a per-row seed and a property test across chunk layouts and both engines. The two validators became one, in Rust, that owns every row, with a sweep that went red the day Polars changed. Not one of them is a comment anymore.
-->

---
layout: statement
---

# A limitation you cannot enforce from inside the engine is not handled.<br>It is a <span class="ox">bug with a delivery date</span>.

Move it inside, or make it red.

<!--
So the rule, whatever your two giants are. A limitation you write down but cannot enforce from inside the engine is not handled. It is a bug with a delivery date. Move it inside, or make it red.
-->

---
layout: section
---

# Borrow the maths. Borrow the engine. Own the contracts.

Every mistake in this talk is written up in the design notes, with the date it shipped and the date it was fixed.

<div class="mt-10">

`pip install polars-stats` · [github.com/FBruzzesi/polars-stats](https://github.com/FBruzzesi/polars-stats) · [fbruzzesi.github.io/polars-stats](https://fbruzzesi.github.io/polars-stats/)

</div>

<!--
Borrow the maths. Borrow the engine. Own the contracts. pip install polars-stats. The design notes have every one of these mistakes written up with the date it shipped and the date it was fixed. Thank you.
-->
