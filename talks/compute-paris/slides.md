---
theme: none
title: Vectorized statistical distributions, built on the Polars engine
info: compute! Paris 2026, 25 minutes plus 5 of Q&A
author: Francesco Bruzzesi
colorSchema: light
codeCopy: false
transition: fade
mdc: true
layout: cover
---

# Vectorized statistical distributions, built on the Polars engine

Francesco Bruzzesi

<!--
Thank you for having me. [pause]

(Timing marks: beat 1 ends 4:00, beat 2 8:30, beat 3 12:45, beat 4 17:00, beat 5 21:30, beat 6 26:00. If beat 2 ends past 9:00, take cut list items 1 to 3.)
-->

---
layout: section
class: divider
---

# Each sensor has its own baseline. How do I simulate readings and score real ones, row by row, on a dataframe?

<!--
Each sensor has its own baseline. How do I simulate readings and score real ones, row by row, on a dataframe? [pause]
-->

---

# Two of these are faults. Which?

<div class="pair">
<table class="frame">
<thead><tr><th>sensor</th><th>reading</th><th v-click="1">mu</th><th v-click="1">sigma</th><th v-click="1">z</th></tr></thead>
<tbody>
<tr><td>temp-a</td><td>9.8</td><td v-click="1">10.0</td><td v-click="1">0.5</td><td v-click="1">-0.4</td></tr>
<tr><td>temp-a</td><td>10.4</td><td v-click="1">10.0</td><td v-click="1">0.5</td><td v-click="1">0.8</td></tr>
<tr><td>temp-a</td><td :class="{ flag: $clicks >= 1 }">40.0</td><td v-click="1">10.0</td><td v-click="1">0.5</td><td v-click="1">60.0</td></tr>
<tr><td>temp-b</td><td>99.4</td><td v-click="1">100.0</td><td v-click="1">2.0</td><td v-click="1">-0.3</td></tr>
<tr><td>temp-b</td><td>101.2</td><td v-click="1">100.0</td><td v-click="1">2.0</td><td v-click="1">0.6</td></tr>
<tr><td>temp-b</td><td :class="{ flag: $clicks >= 1 }">250.0</td><td v-click="1">100.0</td><td v-click="1">2.0</td><td v-click="1">75.0</td></tr>
</tbody>
</table>
<img v-click="2" class="bells" src="/bells.svg" alt="One bell curve per sensor, with the upper tail beyond a healthy reading shaded and each fault far off the chart">
</div>

<p v-click="1" class="caption">z = (reading - mu) / sigma: how many standard deviations from its own sensor's mean</p>

<!--
Here are six temperature readings from two sensors. Sensor A usually sits around 10 degrees, give or take half a degree. Sensor B sits around 100, give or take 2. Two of these readings are faults. Which ones? [pause]

[click] Forty on sensor A, two hundred and fifty on sensor B. You just did that in your head with a z-score: how many standard deviations a reading sits from its sensor's mean. Forty is sixty standard deviations above ten.

[click] That tells you how far. What I usually need is how surprising: if the sensor is healthy, how likely is a reading at least this high? That's the tail of the distribution, and it lets you set an alarm at "one in a thousand" instead of at a magic number of degrees, different for every sensor.

And before I score real readings, I want fake ones: simulated telemetry for each sensor, from its own baseline, one draw per row or a thousand per row. So, two jobs: sample and score. And every row has its own distribution.

I'm Francesco. One thing that matters for the next 25 minutes: I'm not a Rust expert, and a good part of the Rust in polars-stats, the library this talk is about, was written with AI assistance. This talk is about what you can build anyway, standing on the right shoulders, and by the end you'll know where your own bugs are going to live.
-->

---

# scipy already scores every row against its own distribution

<p class="lede">Polars: describe the whole query first, and Polars plans it before running any of it.</p>

<table class="vocab">
<thead><tr><th><code>pdf</code></th><th><code>cdf</code></th><th><code>sf</code></th><th><code>ppf</code> · <code>isf</code></th><th><code>rvs</code></th></tr></thead>
<tbody><tr><td>density</td><td>probability below a value</td><td>probability above it</td><td>from a probability back to a value</td><td>draw samples</td></tr></tbody>
</table>

<div v-click="1">

```python {all|1,5,6}{at:2}
df = lf.collect()                          # the lazy query ends here
mu, sigma, x = (df[c].to_numpy() for c in ("mu", "sigma", "reading"))
rng = np.random.default_rng(42)
df = df.with_columns(
    simulated=pl.Series(stats.norm(loc=mu, scale=sigma).rvs(random_state=rng)),
    upper_tail=pl.Series(stats.norm(loc=mu, scale=sigma).sf(x)),
)
```

</div>

<div v-click="2" class="tags"><span class="tag">the query ends</span><span class="tag">alignment is yours</span><span class="tag">?</span></div>

<p class="footnote">scipy: Python's scientific library, since 2001 · <code>scipy.stats</code>: more than a hundred distributions, one vocabulary · polars-stats copies it word for word</p>

<!--
My data lives in Polars, a dataframe library written in Rust, with a Python API. Its trick is laziness: you describe the whole query first, and Polars plans it before running any of it.

The first pair of shoulders is scipy, Python's scientific library, around since 2001. Its statistics module, `scipy.stats`, has more than a hundred distributions, and they all speak one vocabulary: `pdf` for the density, `cdf` for the probability below a value, `sf`, the survival function, for the probability above it, `ppf` and `isf` to go from a probability back to a value, and `rvs` to draw samples. polars-stats copies that vocabulary, word for word.

And scipy already solves my problem. Give it an array of means and an array of standard deviations, and it scores every element against its own distribution, in compiled code, with no Python loop. So the honest first attempt is this one. [click] Take the columns out, give them to scipy, put the results back. It works. I did it for years.

What it costs is where the result lands. [click] First, the query: to leave a lazy query you have to run it, so everything after this line is a second query, and the optimiser never sees the two together. Second, alignment: the result is a bare array, and keeping it on the right rows through the next join is your job. Third, a quiz.
-->

---

# `stats.norm(loc=0, scale=-1).sf(1.0)` returns what?

<p class="lead">A normal distribution with a standard deviation of minus one.</p>

<div v-click="1" class="reveal">nan</div>

<p v-click="1" class="footnote center">no warning emitted · scipy 1.18.1</p>

<!--
A normal distribution with a standard deviation of minus one. That's not a distribution. What does scipy give you for its tail? [pause]

[click] `nan`. No warning. A modelling error, travelling down your pipeline dressed up as missing data.

None of this is scipy's fault: an array has no query plan, no row identity and no null.
-->

---

# I wanted the answer to stay inside the query

<div class="spine">
<div class="slot">
<p class="slot-label">inside the query</p>
<table class="frame">
<thead><tr><th>sensor</th><th>reading</th><th>mu</th><th>sigma</th><th>z</th><th class="new">upper_tail</th></tr></thead>
<tbody>
<tr><td>temp-a</td><td>9.8</td><td>10.0</td><td>0.5</td><td>-0.4</td><td>?</td></tr>
<tr><td>temp-a</td><td>10.4</td><td>10.0</td><td>0.5</td><td>0.8</td><td>?</td></tr>
<tr><td>temp-a</td><td>40.0</td><td>10.0</td><td>0.5</td><td>60.0</td><td>?</td></tr>
<tr><td>temp-b</td><td>99.4</td><td>100.0</td><td>2.0</td><td>-0.3</td><td>?</td></tr>
<tr><td>temp-b</td><td>101.2</td><td>100.0</td><td>2.0</td><td>0.6</td><td>?</td></tr>
<tr><td>temp-b</td><td>250.0</td><td>100.0</td><td>2.0</td><td>75.0</td><td>?</td></tr>
</tbody>
</table>
</div>
<div class="stack side">
<div class="giant"><b>scipy</b>the vocabulary, and the ruler</div>
</div>
</div>

<!--
But I wanted the answer to stay inside the query.
-->

---
layout: section
class: divider
---

# Can the distribution live inside the Polars query instead?

<!--
Polars can do the first half on its own.
-->

---

# Polars computes the z-score on every core, and has no `erf`

```python
z = (pl.col("reading") - pl.col("mu")) / pl.col("sigma")
```

<div class="pair">
<table class="frame">
<thead><tr><th>sensor</th><th>reading</th><th>mu</th><th>sigma</th><th>z</th></tr></thead>
<tbody>
<tr><td>temp-a</td><td>9.8</td><td>10.0</td><td>0.5</td><td>-0.4</td></tr>
<tr><td>temp-a</td><td>10.4</td><td>10.0</td><td>0.5</td><td>0.8</td></tr>
<tr><td>temp-a</td><td>40.0</td><td>10.0</td><td>0.5</td><td>60.0</td></tr>
<tr><td>temp-b</td><td>99.4</td><td>100.0</td><td>2.0</td><td>-0.3</td></tr>
<tr><td>temp-b</td><td>101.2</td><td>100.0</td><td>2.0</td><td>0.6</td></tr>
<tr><td>temp-b</td><td>250.0</td><td>100.0</td><td>2.0</td><td>75.0</td></tr>
</tbody>
</table>
<div>
<div class="formula">upper_tail = ½ · (1 − <span class="alarm">erf</span>(z / √2)) <span class="alarm note">Polars: no <code>erf</code></span></div>
<div v-click="1" class="attempt"><code>map_elements</code>: one row at a time, through the interpreter, on one thread</div>
<div v-click="1" class="attempt">a textbook <code>erf</code> in Polars arithmetic: fine in the middle, wrong in the tails</div>
<div v-click="2" class="attempt move">an expression plugin: a compiled Rust function that Polars calls as one of its own expressions; the maths is still yours</div>
</div>
</div>

<!--
Reading minus mu, divided by sigma: plain arithmetic, on every core. The next step is the problem. Turning a z-score into a tail probability needs a special function, the error function, `erf`, and Polars doesn't have one. Fair enough: it's a dataframe engine, not a statistics package.

So what would you try? [pause] [click] Call a Python function once per row with `map_elements`: it works, one row at a time, through the interpreter, on one thread. Type a textbook approximation of `erf` as Polars arithmetic: fine in the middle, wrong in the tails, which is exactly where anomaly detection lives.

[click] The third option is the one Polars offers itself: an expression plugin. You write a function in Rust, compile it, and Polars calls it as if it were one of its own expressions. The function receives columns and returns a column, and that is the whole contract.

That leaves the maths, and I was not going to write an error function.
-->

---

# statrs brings the maths, Polars brings the engine

<div class="spine">
<div class="stack">
<div class="giant"><b>Polars</b>the engine</div>
<div class="mine"><b>polars-stats</b></div>
<div class="giant"><b>statrs</b>the maths<span class="detail"><code>Normal::new(mu, sigma)</code>: density, cdf, survival function, inverse cdf, log density, sampling, with <code>erf</code>, gamma and beta underneath</span></div>
</div>
<div class="stack side">
<div class="giant"><b>scipy</b>the vocabulary, and the ruler</div>
</div>
</div>

<p class="footnote">statrs: a Rust crate (Rust's word for a library), roughly <code>scipy.stats</code> for Rust, maintained by volunteers</p>

<!--
Meet the second pair of shoulders: statrs. It's a Rust crate, which is Rust's word for a library, and it's roughly `scipy.stats` for Rust. Build `Normal::new(mu, sigma)` and you get the density, the cdf, the survival function, the inverse cdf, the log density and sampling, with `erf`, gamma and beta underneath. It's maintained by volunteers, and it's the reason polars-stats exists.
-->

---

# Python builds the expression, Rust walks the rows, statrs does the maths

<div class="pair">
<div>
<p class="layer-label">Python: returns a <code>pl.Expr</code>, computes nothing</p>

```python
baseline = ps.Normal(mu="mu", sigma="sigma")
lf.with_columns(upper_tail=baseline.sf("reading"))
```

<p class="layer-label">Rust: columns in, column out</p>

```rust
#[polars_expr(output_type=Float64)]
fn normal_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, sf_value)
}

fn sf_value(dist: &Normal, v: f64) -> Option<f64> {
    Some(dist.sf(v))              // statrs: the maths
}
```

</div>
<div v-click="1">
<p class="layer-label">registered with Polars</p>

```python {5}
register_plugin_function(
    plugin_path=LIB,
    function_name="normal_sf",
    args=[x, mu, sigma],
    is_elementwise=True,
)
```

<p class="promise">a promise: row i of the output depends only on row i of the inputs</p>
</div>
</div>

<!--
So here's the whole anatomy. On top, a Python class whose parameters can be column names. Its methods compute nothing: each one returns a Polars expression that says "call this Rust function on these columns". In the middle, the Rust function: it gets the columns, walks the rows, builds a distribution per row. At the bottom, one line: `dist.sf(v)`. That line is statrs.

It wasn't always this plain: the first versions generated these functions with Rust macros, code that writes code, and I couldn't review what they produced. When you're not an expert in a language, code you can read beats code that saves typing.

Now the part I enjoy the most. [click] When I register the function with Polars, I pass one flag: `is_elementwise=True`. It's a promise: row i of the output depends only on row i of the inputs. In exchange for that one promise, Polars gives me everything else. Watch.
-->

---

# Polars plans the whole query before it runs a single row

````md magic-move {at:3}
```python
scored = (
    pl.scan_parquet("readings.parquet")
    .with_columns(upper_tail=ps.Normal(mu="mu", sigma="sigma").sf("reading"))
    .filter(pl.col("sensor") == "temp-b")
    .filter(pl.col("upper_tail") < 1e-3)
)
print(scored.explain())
```
```python
in_memory = scored.collect(engine="in-memory")
streaming = scored.collect(engine="streaming")
in_memory.height, streaming.height, in_memory.equals(streaming)
# (482, 482, True)
```
````

<div v-click="1" class="plan">

```text {all|7}{at:2}
FILTER col("upper_tail") < 0.001
FROM
   WITH_COLUMNS:
   [col("reading").…/_internal.abi3.so:normal_sf([col("mu"), col("sigma")]).alias("upper_tail")]
    Parquet SCAN [readings.parquet]
    PROJECT */4 COLUMNS
    SELECTION: col("sensor") == "temp-b"
```

<span v-click="2" class="callout">the filter moved into the file reader</span>
</div>

<p v-click="4" class="engines"><code>sink_parquet</code>: file to file · <code>over</code> / <code>group_by</code>: per group · all 10 cores</p>

<p class="footnote">Polars 1.44.2 · a 1,000,000-row Parquet file of the two sensors</p>

<!--
Here's a pipeline. Scan a Parquet file of a million readings, add the tail probability, keep sensor B, keep the readings rarer than one in a thousand. [click] And here's what Polars plans to do, read from the bottom up. [pause] [click] The filter on sensor B, which I wrote after the scoring, has moved down into the file reader. Half the rows are never decoded, so they never reach my function. The filter on the score stays on top, because it needs the score. That's called predicate pushdown, and I wrote none of it.

[click] Run it on the in-memory engine, and on the streaming engine, which reads the file in small batches called morsels, so the data never has to fit in memory. Same 482 rows, identical. [click] Swap `collect` for `sink_parquet` and it streams from file to file. Put it inside a `group_by` or an `over` and it works per group. All of it on Polars' thread pool, across every core.

I wrote a loop over rows. Polars made it lazy, optimised, parallel, streaming and groupable. statrs made the numbers right. scipy told me what right looks like.
-->

---

# polars-stats is the thin strip between the engine and the maths

<div class="spine">
<div class="stack">
<div class="giant"><b>Polars</b>the engine: lazy, optimised, parallel, streaming, groupable</div>
<div class="mine"><b>polars-stats</b></div>
<div class="giant"><b>statrs</b>the maths: the numbers, right</div>
</div>
<div class="stack side">
<div class="giant"><b>scipy</b>the vocabulary, and the ruler</div>
</div>
</div>

<table class="frame spine-frame">
<thead><tr><th>sensor</th><th>reading</th><th>z</th><th class="new">upper_tail</th></tr></thead>
<tbody>
<tr><td>temp-a</td><td>9.8</td><td>-0.4</td><td>0.655422</td></tr>
<tr><td>temp-a</td><td>10.4</td><td>0.8</td><td>0.211855</td></tr>
<tr><td>temp-a</td><td>40.0</td><td>60.0</td><td>0.0</td></tr>
<tr><td>temp-b</td><td>99.4</td><td>-0.3</td><td>0.617911</td></tr>
<tr><td>temp-b</td><td>101.2</td><td>0.6</td><td>0.274253</td></tr>
<tr><td>temp-b</td><td>250.0</td><td>75.0</td><td>0.0</td></tr>
</tbody>
</table>

<!--
So the picture is this: Polars on top, statrs at the bottom, scipy beside them holding the ruler. And between Polars and statrs, a thin strip of code that I actually wrote. That strip is polars-stats.
-->

---
layout: section
class: divider
---

# If the engine and the maths are borrowed, what is left for me to write?

<!--
The glue started very small.
-->

---

# The first commit seeded one generator and walked the rows in order

<p class="lede">Bernoulli, a biased coin · one method, <code>sample</code> · under seventy lines of Rust</p>

```rust
let mut rng = ChaCha20Rng::seed_from_u64(seed);
for i in 0..n {
    let dist = Bernoulli::new(p[i])?;
    values.push(dist.sample(&mut rng));
}
```

<div v-click="1">

```python
"""... Reproducibility under ``seed`` assumes a
single-chunk input series; chunked / streaming inputs are not
supported.
"""
```

</div>

<p class="footnote">first commit, 26 April 2026</p>

<!--
26 April 2026, the first commit of polars-stats. One distribution, Bernoulli: a biased coin that comes up 1 with probability p. One method, `sample`. Under seventy lines of Rust. The idea is what anyone would write: take the seed, start one random number generator, walk the rows in order, and draw one value per row with statrs.

And the docstring was honest. [click] "Reproducibility under seed assumes a single-chunk input series; chunked or streaming inputs are not supported."

Not supported. As if that were a mode the caller could switch off.
-->

---

# Polars hands a plugin pieces, not the column

<div class="pieces">
<div class="lane whole"><span>the column</span><div class="bar"><span class="piece"></span></div></div>
<div class="lane"><span>chunks</span><div class="bar"><span class="piece"></span><span class="piece"></span><span class="piece"></span></div></div>
<div class="lane"><span>threads</span><div class="bar"><span class="piece"></span><span class="piece"></span><span class="piece"></span><span class="piece"></span></div></div>
<div class="lane"><span>morsels (streaming)</span><div class="bar"><span class="piece"></span><span class="piece"></span><span class="piece"></span><span class="piece"></span><span class="piece"></span><span class="piece"></span><span class="piece"></span><span class="piece"></span></div></div>
</div>

<p class="caption">↺ 42: the generator restarts from seed 42 at the start of every piece</p>

<p v-click="1" class="predict">Same frame, same seed, in-memory vs streaming. Same numbers?</p>

<p v-click="2" class="answer"><b class="alarm">No</b>: the seed reproduced the chunk layout, not the query.</p>

<!--
Polars doesn't hand a plugin "the column". It hands it pieces. A column can live in several chunks, the thread pool splits work across cores, the streaming engine sends morsels, a group by sends one group at a time. My function runs once per piece, and every call restarts the generator from the same seed.

[click] So, a prediction. [pause] Same frame, same seed, in-memory engine against streaming engine. Same numbers? [pause] [click] No. The first rows of every piece repeat each other, and where the pieces start depends on the engine, not on your data. The seed was reproducing the chunk layout, not the query.

That one never reached a release, but it changed how I think about the plugin. It sees whatever pieces Polars decides to give it.
-->

---

# Every row draws from its own generator, keyed on seed and row index

<div class="pair wide-left">
<div>

````md magic-move {at:1}
```rust
let mut rng = ChaCha20Rng::seed_from_u64(seed);
for i in 0..n {
    let dist = Bernoulli::new(p[i])?;
    values.push(dist.sample(&mut rng));
}
```
```rust
for i in 0..n {
    let mut rng = row_rng(seed, index[i]);
    let dist = Bernoulli::new(p[i])?;
    values.push(dist.sample(&mut rng));
}
```
````

<p v-click="1" class="caption">the row index is an ordinary input column, <code>pl.int_range(0, pl.len())</code>, so Polars carries it through chunks, morsels and groups</p>
</div>
<div>
<table class="frame dice">
<thead><tr><th>sensor</th><th>reading</th><th v-click="1">row's die</th></tr></thead>
<tbody>
<tr><td>temp-a</td><td>9.8</td><td v-click="1">(42, 0)</td></tr>
<tr><td>temp-a</td><td>10.4</td><td v-click="1">(42, 1)</td></tr>
<tr><td>temp-a</td><td>40.0</td><td v-click="1">(42, 2)</td></tr>
<tr class="cut"><td>temp-b</td><td>99.4</td><td v-click="1">(42, 3)</td></tr>
<tr><td>temp-b</td><td>101.2</td><td v-click="1">(42, 4)</td></tr>
<tr><td>temp-b</td><td>250.0</td><td v-click="1">(42, 5)</td></tr>
</tbody>
</table>
<p v-click="1" class="caption">cut the pieces anywhere: every row keeps its die</p>
</div>
</div>

<!--
So the move: stop thinking of one stream walking down the rows. [click] Give every row its own generator, seeded from two numbers, the seed and the row's index. Row 7 draws from the stream for 42 and 7, whichever piece, thread or engine it lands in. And the row index is an ordinary input column, so Polars carries it through chunks, morsels and groups for me.

That fix came with a bill.
-->

---

# Per-row generators have to be cheap to build

<div class="bill">
<p><span v-mark.strike-through.red="1">one <code>ChaCha20</code> per row: 10 to 20 times slower</span></p>
<p v-click="1"><code>Pcg64Mcg</code>: about 5.9 s to 0.29 s on 1M rows × 100 draws</p>
<p v-click="1"><code>samples</code>: a row's draws are the first values of that row's own stream</p>
</div>

<div v-click="2" class="aside">
<div>statrs Binomial: n coin flips per draw, O(n)</div>
<div><code>rand_distr</code>: cost flat in n</div>
</div>

<p v-click="3" class="today">Today a seeded column is identical across runs, chunk layouts, thread counts and both engines. CI runs every test once per engine.</p>

<p class="footnote"><code>Pcg64Mcg</code> numbers: that day's benchmark, 1 June 2026</p>

<!--
My generator, ChaCha20, is a cryptographic one, and building one per row made sampling ten to twenty times slower. [click] The fix for the fix was `Pcg64Mcg`, which costs a handful of integer operations to build. On that day's benchmark, a million rows times a hundred draws went from about 5.9 seconds to 0.29. And `samples`, a thousand draws per row, is the first thousand values of that row's own stream.

[click] One place where a giant said no: statrs draws a Binomial by flipping n coins, so n random numbers per draw. Binomial sampling borrows from another crate instead, `rand_distr`, whose cost doesn't grow with n. Everything else about the Binomial still comes from statrs.

[click] Today a seeded column repeats across runs, chunk layouts, thread counts and both engines, and CI runs every test once per engine.
-->

---

# A seeded column should belong to the data, not to the chunk layout

<table class="compare">
<thead><tr><th><code>seed=42</code>, 1M rows, Polars 1.44.2</th><th>re-run</th><th>in-memory vs streaming</th><th>1 chunk vs 4 chunks</th></tr></thead>
<tbody>
<tr><td>polars_rng</td><td colspan="3">no <code>seed</code> argument</td></tr>
<tr><td>polars-random 0.5.0</td><td>same</td><td>different</td><td>different (column parameters)</td></tr>
<tr><td>polars-stats 0.1.0</td><td>same</td><td>same</td><td>same</td></tr>
</tbody>
</table>

<p class="caption">polars-random and polars_rng are good sampling-first plugins, and polars_rng's sampling catalogue is wider than polars-stats'</p>

<p class="footnote">measured 2026-10-03</p>

<!--
And this turned out to be the one thing I couldn't find anywhere else. There are two good Polars plugins for sampling, polars-random and polars_rng, and for pure simulation polars_rng's catalogue is wider than polars-stats'. But polars_rng takes no seed at all. polars-random takes one, and on its latest release the same seed gives one column on the in-memory engine and a different one on streaming. For simulation, that's a reasonable trade. For testing a pipeline, I needed the seeded column to belong to the data, not to how Polars happened to cut it.
-->

---

# The first promise neither giant made: order

<div class="spine">
<div class="stack">
<div class="giant"><b>Polars</b>the engine</div>
<div class="mine"><b>polars-stats</b><div class="pins"><span class="grown">order</span></div></div>
<div class="giant"><b>statrs</b>the maths</div>
</div>
<div class="stack side">
<div class="giant"><b>scipy</b>the vocabulary, and the ruler</div>
</div>
</div>

<table class="frame spine-frame">
<thead><tr><th>sensor</th><th>reading</th><th>z</th><th>upper_tail</th><th class="new">simulated</th></tr></thead>
<tbody>
<tr><td>temp-a</td><td>9.8</td><td>-0.4</td><td>0.655422</td><td class="todo">TODO</td></tr>
<tr><td>temp-a</td><td>10.4</td><td>0.8</td><td>0.211855</td><td class="todo">TODO</td></tr>
<tr><td>temp-a</td><td>40.0</td><td>60.0</td><td>0.0</td><td class="todo">TODO</td></tr>
<tr><td>temp-b</td><td>99.4</td><td>-0.3</td><td>0.617911</td><td class="todo">TODO</td></tr>
<tr><td>temp-b</td><td>101.2</td><td>0.6</td><td>0.274253</td><td class="todo">TODO</td></tr>
<tr><td>temp-b</td><td>250.0</td><td>75.0</td><td>0.0</td><td class="todo">TODO</td></tr>
</tbody>
</table>

<p class="footnote">&lt;TODO: simulated column: record <code>ps.Normal(mu="mu", sigma="sigma").sample(seed=42)</code> on the six-row frame&gt;</p>

<!--
First pin on the strip: order.
-->

---
layout: section
class: divider
---

# The plugin sees pieces. Do the inputs arrive at full length, and does every row run?

<!--
Here's another comment from that first commit.
-->

---

# Polars does not stretch a one-value input before the plugin runs

<blockquote class="quote"><code>is_elementwise=True</code> does NOT broadcast inputs before the plugin runs … a length-1 <code>pl.lit(p)</code> would make the plugin draw a single sample which polars then repeats across the whole frame.</blockquote>

<p class="footnote">first commit, 26 April 2026</p>

<div v-click="1" class="ways">
<div class="way ok"><code>0.5</code><span>padded in Python</span></div>
<div class="way bad"><code>pl.lit(0.5)</code><span>one row long only at run time: <b class="alarm">Python cannot see this</b></span></div>
<div class="way bad"><code>pl.col("reading").mean()</code><span>one row long only at run time: <b class="alarm">Python cannot see this</b></span></div>
</div>

<!--
Roughly: Polars doesn't broadcast inputs into a plugin, so a one-value `pl.lit(p)` would draw a single sample that Polars then repeats across the frame. Broadcasting means stretching one value to the length of the column. So Python stretched every plain number before calling Rust. Handled.

It handled the case I thought of. [click] But Polars also lets you write `pl.lit(0.5)`, or a column's mean, or its first value. Each is one row long, and only the running engine knows it, not the Python that builds the query. And the Rust helper that walks several columns together stops at the shortest one.

Prediction time.
-->

---

# The right answer has 1,000,000 distinct values. How many do we get?

<p class="lede">PyPI 0.0.1 · one million rows</p>

```python
Normal(mu=0.0, sigma=pl.lit(0.5)).cdf("x")
```

<div v-click="1" class="tiles">
<div class="tile"><span class="label">in-memory</span><span class="num alarm">1</span></div>
<div class="tile"><span class="label">streaming</span><span class="num alarm">10</span></div>
</div>

<p v-click="1" class="caption center">height 1,000,000 on both · no error</p>

<!--
Version 0.0.1, straight from PyPI. A million rows, a Normal with sigma equal to `pl.lit(0.5)`, scored with `cdf`. The right answer has a million distinct values. How many do we get? [pause]

[click] In-memory: one. Row zero's answer, a million times. Streaming: ten, one per morsel. The right height, no error, no warning. That one I published.
-->

---

# Alignment by length belongs in Rust

````md magic-move {at:1}
```python
if isinstance(p, pl.Expr):
    self._p = p                             # passed through as is
elif isinstance(p, float):
    self._p = pl.repeat(p, n=pl.len())      # padded in Python
```
```rust
// align_inputs: the first thing every plugin does
for input in inputs {
    if input.len() == 1 { /* stretch it to the others' length */ }
    else if input.len() != anchor.len() { /* raise ShapeMismatch */ }
}
```
````

<p v-click="1" class="caption">by length at run time, not by kind of expression</p>

<div v-click="2">

```python
r = pl.col("reading")
lf.with_columns(upper_tail=ps.Normal(mu=r.mean(), sigma=r.std()).sf(r).over("sensor"))
```

<p class="caption"><code>r.mean()</code> is one value per group: length 1, stretched by the plugin</p>
</div>

<div v-click="3" class="stack mini"><div class="mine"><b>polars-stats</b><div class="pins"><span class="grown">order</span><span class="grown">length</span></div></div></div>

<p class="footnote">alignment in Rust: breaking release 0.0.2 · the <code>over</code> line on 0.0.1: <code>ShapeError</code> on the in-memory engine; today: both engines</p>

<!--
The move: align in Rust, not in Python. [click] The first thing every plugin does now is stretch any input of length one to the others' length, and raise if two real lengths disagree, judged by actual length at run time, not by kind of expression. It changed what some calls return, so it shipped as a breaking release, 0.0.2.

[click] That fix is what makes this line work. It estimates each sensor's baseline from its own readings and scores every reading against it, in one expression, inside an `over`. The mean and the standard deviation are one value per sensor: length one. The plugin stretches them, Polars does the grouping. On 0.0.1 this line raised a shape error on the in-memory engine. Today it runs on both. [click] Second pin: length.
-->

---

# The validator sat inside Polars' if-else

<p class="lede">A Polars expression cannot raise on a bad row, so <code>validate(params)</code> was a tiny Rust plugin that only checks and raises.</p>

<div v-click="1" class="formula"><code>when(null).then(null).otherwise(validate(params) → formula)</code></div>

<p v-click="2" class="callout">Polars 1.44: each branch is evaluated only on the rows that select it</p>

<p v-click="3" class="caption">same published wheel (0.0.1), same frame: Uniform, <code>min = 1</code>, <code>max = 0</code>, <code>x = -5</code></p>

<div v-click="3" class="tiles">
<div class="tile"><span class="label">Polars 1.43.2</span><span class="num word">ComputeError</span></div>
<div v-click="4" class="tile"><span class="label">Polars 1.44.2</span><span class="num alarm">0.0</span></div>
</div>

<!--
Now, which rows run. Remember scipy's `nan` for a negative standard deviation? I wanted an error instead. But a Polars expression can't raise on a bad row. So for the methods simple enough to write as Polars arithmetic, the parameters went through a tiny Rust plugin whose only job was to check and raise. [click] And that check sat inside a `when / then / otherwise`, Polars' if-else, next to the null handling.

[click] Then Polars 1.44 shipped a good optimisation: evaluate each branch only on the rows that select it. For Polars' own expressions, that's pure win. For my validator, it meant it no longer saw the rows it existed to reject.

[click] Same published wheel, same three-row frame: a Uniform whose maximum is below its minimum. On Polars 1.43: an error, as designed. On Polars 1.44: [pause] [click] zero. A quiet number where an error should be. And a plain `pip install` would give you exactly that pair.
-->

---

# About ten days from the cap to deleting the wrapper

<div class="steps">
<div>cap <code>polars&lt;1.44</code></div>
<div>move the closed forms into Rust</div>
<div>lift the cap</div>
<div>delete the Python null wrapper</div>
</div>

<div class="spine">
<div class="stack">
<div class="giant"><b>Polars</b>the engine</div>
<div class="mine"><b>polars-stats</b><div class="pins"><span class="grown">order</span><span class="grown">length</span><span v-click="1" class="grown">every row</span></div></div>
<div class="giant"><b>statrs</b>the maths</div>
</div>
<div class="stack side">
<div class="giant"><b>scipy</b>the vocabulary, and the ruler</div>
</div>
</div>

<p class="footnote">upstream: Polars issue #29005, open</p>

<!--
The fix took about ten days. Cap Polars below 1.44, so nobody upgrades into it. Move every closed form that branches on the value into Rust, so the check runs on every row by construction. Lift the cap. Then find the same leak one level up, in a Python null wrapper, and delete it. I opened an issue upstream too; it's still open, and that's fine, because the fix belongs on my side.

Polars never promised that every branch sees every row. I assumed it. [click] Third pin: every row. The plugin owns its rows now.
-->

---
layout: section
class: divider
---

# Every row runs now. Is the number on each row right?

<!--
Every row runs. Is the number on each row right?

The natural answer: check against scipy.
-->

---

# Every method is checked against scipy, and all of it is green

<div class="spine">
<div class="stack">
<div class="giant"><b>Polars</b>the engine</div>
<div class="mine"><span v-click="1" class="belief">safe (I wrote it)</span><b>polars-stats</b><div class="pins"><span class="grown">order</span><span class="grown">length</span><span class="grown">every row</span></div></div>
<div class="giant"><span v-click="1" class="belief">risky (I can't read it)</span><b>statrs</b>the maths</div>
</div>
<div class="stack side">
<div class="giant"><b>scipy</b>the vocabulary, and the ruler<span class="detail"><span class="ox">✓</span> parity on every method<br><span class="ox">✓</span> property tests (hypothesis)</span></div>
</div>
</div>

<div v-click="2" class="rulers">
<div class="ruler"><span>scipy, float64</span><div class="track"><div class="bar short"></div><code>1e-308</code></div></div>
<div class="ruler"><span>polars-stats, float64</span><div class="track"><div class="bar short"></div><code>1e-308</code></div></div>
<div v-click="3" class="ruler"><span>mpmath, 50 digits</span><div class="track"><div class="bar long"></div></div></div>
</div>

<!--
Every method is tested against scipy on the same inputs, plus property tests: a tool called hypothesis invents thousands of inputs and checks what must always hold, like a cdf staying between zero and one and never going down.

All green. [click] At that point I believed the risky code was the code I couldn't read, the special functions inside statrs, because those are the hard parts. The algebra I'd written myself, one minus this, divided by that, was the safe part. I could read it.

[click] But scipy and polars-stats both compute in 64-bit floating point. [pause] In the far tails both run out of digits, and two zeros agree perfectly. Parity can't tell you who's right where everybody saturates.

[click] So I brought a longer ruler: mpmath, a Python library that computes with as many digits as you ask for. Fifty, here, across every method and distribution, with parameters many orders of magnitude apart.
-->

---

# I budgeted time for three defects

<div class="tiles">
<div class="tile"><span class="label">budgeted</span><span class="num">3</span></div>
<div v-click="1" class="tile"><span class="label">found</span><span class="num alarm">11</span></div>
</div>

<div v-click="2" class="breakdown center">
<p>Bernoulli + Uniform: 4 · Exponential: 3</p>
<p>Nine of eleven: one-to-three-line fixes in closed forms I wrote</p>
</div>

<p class="footnote">oracle: mpmath at 50 digits</p>

<!--
I'd budgeted time for three defects.

[click] It found eleven. [pause] And not where I expected. [click] Bernoulli and Uniform, the two trivial distributions, produced four of them. Exponential produced three more. Nine of the eleven were one-to-three-line fixes, in closed forms I'd written myself.
-->

---

# A coin that comes up 1 with probability `1e-300`

```python
Bernoulli(p=1e-300).sf(0.5)
```

<div class="derive">
<div v-click="1"><span>should be</span><code>1e-300</code></div>
<div v-click="2"><span>returned</span><code class="alarm">0.0</code></div>
<div v-click="3"><span>computed as</span><code>1 - (1 - p)</code></div>
<div v-click="4"><span>in float64</span><code>1 - 1e-300 == 1.0</code></div>
</div>

<p v-click="5" class="callout">compute it directly · standardise last, not first · in the tails, work in logarithms</p>

<!--
My favourite. A Bernoulli coin with p equal to ten to the minus three hundred. What's the probability of a value above one half? [click] It's p, a perfectly representable number. [click] I returned zero. [click] I computed it as one minus the cdf, and the cdf is one minus p. [click] In 64-bit floats, one minus ten to the minus three hundred is exactly one. The p was gone before the subtraction even happened.

[click] The fixes were small: compute what you want directly, not as the difference of two nearly equal numbers; standardise last, not first; in the tails, work in logarithms.
-->

---

# `log_sf` ranks the faults that `sf` rounds to zero

<table class="frame">
<thead><tr><th>sensor</th><th>reading</th><th>z</th><th>upper_tail</th><th v-click="1" class="new">log_tail</th></tr></thead>
<tbody>
<tr><td>temp-a</td><td>9.8</td><td>-0.4</td><td>0.655422</td><td v-click="1">-0.422476</td></tr>
<tr><td>temp-a</td><td>10.4</td><td>0.8</td><td>0.211855</td><td v-click="1">-1.551851</td></tr>
<tr><td>temp-a</td><td>40.0</td><td>60.0</td><td class="flag">0.0</td><td v-click="1" class="flag">-1805.013561</td></tr>
<tr><td>temp-b</td><td>99.4</td><td>-0.3</td><td>0.617911</td><td v-click="1">-0.48141</td></tr>
<tr><td>temp-b</td><td>101.2</td><td>0.6</td><td>0.274253</td><td v-click="1">-1.293704</td></tr>
<tr><td>temp-b</td><td>250.0</td><td>75.0</td><td class="flag">0.0</td><td v-click="1" class="flag">-2817.736604</td></tr>
</tbody>
</table>

<p class="caption">true tails of the two faults: about <code>1e-784</code> and <code>1e-1224</code>, both below what a float64 can hold</p>

<p v-click="1" class="caption">sort on <code>log_tail</code>: the <code>temp-b</code> fault comes first</p>

<p class="footnote">values from the docs tutorial, step 5</p>

<!--
Back to our sensors: the true tails of the two faults are around ten to the minus 784 and ten to the minus 1224. `sf` says zero for both, so you can't rank them. [click] `log_sf` says minus 1805 and minus 2817, and the worse fault sorts first.
-->

---

# `Beta(s, s).cdf(0.5)` should be exactly 0.5

<div class="tiles">
<div v-click="1" class="tile"><span class="label">s = 1e6</span><span class="num">0.4912</span></div>
<div v-click="1" class="tile"><span class="label">s = 1e7</span><span class="num">0.2129</span></div>
<div v-click="2" class="tile"><span class="label">s = 1e8</span><span class="num alarm">-1.147</span></div>
</div>

<p v-click="3" class="callout">upstream, as an issue with a reproducer: the input · the 50-digit answer · what statrs returns</p>

<p class="footnote">statrs 0.19.1</p>

<!--
And some limits really were in statrs. A Beta distribution with two equal shapes is symmetric, so its cdf at one half is exactly one half. [click] At shapes of a million, statrs 0.19 says 0.49. At a hundred million, it says [pause] [click] minus 1.147. A probability below zero.

I could have patched that inside polars-stats and moved on. That fixes it for one project. [click] Instead, each limit went upstream as an issue with a reproducer: the input, the 50-digit answer, and what statrs returns. About ten reports.
-->

---

# A good bug report is a contribution

<div class="cards">
<div>
<h3>fixed and released in statrs 0.19.1 · 3</h3>
<div class="card"><code>Beta.sf</code> at tiny <code>x</code></div>
<div class="card"><code>Beta.pdf</code> at 0</div>
<div class="card"><code>Beta.ln_pdf</code> near 1</div>
<p class="note">one with my own PR folded into a contributor's</p>
</div>
<div v-click="1">
<h3>fixed and merged, awaiting release · 2</h3>
<div class="card"><code>Beta.cdf</code> at large shapes</div>
<div class="card"><code>LogNormal.pdf</code> left tail</div>
<p class="note">both fixed by other contributors</p>
</div>
<div v-click="2">
<h3>open · 5</h3>
<div class="card">Beta's inverse cdf, extreme lower tail</div>
<div class="card"><code>Binomial.entropy</code> at <code>n = u64::MAX</code></div>
<div class="card">three Cauchy tails</div>
</div>
</div>

<p class="footnote">status as of 2026-10-03; open ones are documented in polars-stats' accuracy page</p>

<!--
Three were fixed and released within days, in statrs 0.19.1, and one of those fixes folded in a small pull request of mine. [click] Two more were fixed by other contributors and are merged, waiting for the next release; that Beta cdf is one of them. [click] The rest are still open, including Beta's inverse cdf deep in the lower tail, which can panic and take the whole query down with it. Until they're fixed, the polars-stats docs list each one, with the regime where it bites and by how much.

I had underrated that part of open source. A good bug report is a contribution: it cost me an evening, and it fixed the maths for everyone who uses statrs, not only for polars-stats.
-->

---

# The fourth promise neither giant made: digits

<div class="spine">
<div class="stack">
<div class="giant"><b>Polars</b>the engine</div>
<div class="mine"><span v-click.hide="1" class="belief">safe (I wrote it)</span><span v-click="1" class="belief">risky (I wrote it)</span><b>polars-stats</b><div class="pins"><span class="grown">order</span><span class="grown">length</span><span class="grown">every row</span><span class="grown">digits</span></div></div>
<div class="giant"><span v-click.hide="1" class="belief">risky (I can't read it)</span><span v-click="1" class="belief">safe (I can't read it)</span><b>statrs</b>the maths</div>
</div>
<div class="stack side">
<div class="giant"><b>scipy</b>the vocabulary, and the ruler</div>
</div>
</div>

<!--
Fourth pin: digits. [click] And the arrows swap. The most dangerous maths in polars-stats was the maths I wrote myself.
-->

---
layout: section
class: divider
---

# Was it worth it, and what does it cost?

<!--
Was it worth it? Before any speed number, here's everything wrong with the measurement.
-->

---
class: dense
---

# Every number that follows comes from one laptop

<div class="disclaimer conditions">
<p><b>Machine</b> Apple M5, 10 cores (4 performance, 6 efficiency), 24 GB, macOS 26.7.1 · on AC power, other apps closed</p>
<p><b>Software</b> Python 3.12.13 · Polars 1.44.2 · scipy 1.18.1 · NumPy 2.5.3 · polars-stats 0.1.0, release build of <code>3d96178</code></p>
<p><b>Polars side</b> streaming engine, default thread pool (10 threads); the timed call includes building and dispatching the query</p>
<p><b>scipy side</b> classic frozen API, <code>norm(loc=..., scale=...)</code> then the method; samplers get a PCG64 <code>Generator</code>; one thread</p>
<p><b>Not measured</b> scipy's newer API (measured earlier on the Normal, column parameters: 1.3 to 1.9 times faster than the frozen API); peak memory</p>
<p><b><code>samples</code></b> in the column regime scipy draws <code>(draws, rows)</code> and transposes a view: that ratio is optimistic for scipy by about one pass over the output</p>
<p><b>Timing</b> median of up to 50 iterations within 10 s per cell (minimum 3), after one warmup · ratio = scipy median / polars-stats median, above 1 means polars-stats is faster</p>
</div>

<p class="footnote">run on 2026-10-03, 11:17 to 11:49; Pareto and Weibull re-run at 11:51 on a quiet machine</p>

<!--
One machine: my laptop. An Apple M5, ten cores, four performance and six efficiency, 24 gigabytes of memory, macOS, plugged in, other applications closed, but still a laptop. Release builds, and every number is a median of repeated timings. Polars 1.44.2 on the streaming engine, with ten threads. scipy 1.18.1 with NumPy, computing on one thread. Both sides sample from a modern PCG generator. On the scipy side I measure the classic API, `norm(loc, scale).sf(x)`, which is what most scipy code out there looks like; scipy's newer API is faster than that, 1.3 to 1.9 times on the Normal when I measured it. And Polars pays for planning the query on every call, which scipy doesn't.
-->

---

# Three ways to pass parameters are three different code paths

```python
Normal(0.0, 1.0)                           # scalar: plain numbers
Normal(pl.col("mu"), pl.col("sigma"))      # column: one value per row
Normal(pl.lit(0.0), pl.lit(1.0))           # broadcast: a one-value expression, stretched
```

<p class="callout">never compare across them</p>

<!--
There are three ways to pass parameters, and they're different code paths on both sides, so never compare across them: plain numbers, one value per row, and a one-value expression that gets stretched.
-->

---
class: dense
---

# With one parameter value per row, large frames favour polars-stats

<div class="chart">
<img src="/bench-column.svg" alt="Dot plots of the scipy to polars-stats time ratio per distribution, column regime, at 10k, 1M and 10M rows, for sample, samples and sf">
<img v-click="1" src="/bench-column-10k.svg" alt="Overlay: the 10k-row cells where scipy is ahead">
</div>

<div v-click="1" class="annot">
<p>Column regime, 10k rows: scipy is ahead on <code>sample</code> in every cell but DiscreteUniform (up to 6.3x). polars-stats pays 0.24 to 0.35 ms per call whatever the distribution; scipy 0.04 to 0.17 ms.</p>
<p>Other regimes, read on their own: at 10k scipy is ahead on every <code>sample</code> cell (up to 6.7x with plain numbers, 25x with one-value expressions); with one-value expressions also at 1M for Uniform (0.89x) and DiscreteUniform (0.67x; 0.94x at 10M).</p>
</div>

<p class="footnote">column regime only · * scipy: <code>randint</code> with per-row bounds runs <code>np.vectorize</code> (one Python call per row); <code>bernoulli</code> draws via NumPy <code>binomial(1, p)</code></p>

<!--
Here's the per-row case, the reason polars-stats exists. Each dot is a distribution, on a log scale; right of the line, polars-stats is faster. At a million rows, one draw per row is 1.2 to 7.6 times faster than classic scipy, ten draws per row 4 to 19 times, and scoring with `sf` 2.3 to 7.4 times. At ten million, a little more. Two dots sit far to the right, at tens and hundreds of times: the discrete uniform, which scipy samples through a Python-level loop when the bounds differ per row, and Bernoulli, which goes through NumPy's general binomial sampler. Don't quote those.

[click] At ten thousand rows, scipy wins nearly every one-draw-per-row cell, by up to six times. It draws ten thousand numbers in about fifty microseconds. polars-stats spends about three hundred building and dispatching the query, whatever the distribution. Large frames amortise that fixed cost; small frames pay it.

So these numbers are polars-stats and Polars' thread pool together, against classic scipy, on this laptop. That's what you get when you use it, so that's what I measured.
-->

---

# polars-stats costs a shorter catalogue and a compiled dependency

<ul class="costs">
<li>a much shorter catalogue than scipy's hundred-plus</li>
<li>no fitting, no multivariate distributions, no statistical tests</li>
<li>where you need those, scipy is the right tool, and the two live happily in one project</li>
<li>a compiled dependency that follows Polars' plugin interface across releases</li>
</ul>

<p v-click="1" class="callout next">next: every univariate distribution statrs supports</p>

<!--
What it costs: a much shorter catalogue than scipy's hundred-plus, and no fitting, no multivariate distributions, no statistical tests. Where you need those, scipy is the right tool, and the two live happily in one project. And a compiled dependency, which has to follow Polars' plugin interface across releases.

[click] What's next: the short-term plan is to roll out every univariate distribution that statrs supports. The glue is in place and tested on both engines, so each new one is mostly a file of one-line shells and a row in a test registry.
-->

---

# Every bug that mattered lived in the thin strip

<div class="spine">
<div class="stack">
<div class="giant"><b>Polars</b>the engine</div>
<div class="mine"><b>polars-stats</b><div class="pins"><span class="grown">order</span><span class="grown">length</span><span class="grown">every row</span><span class="grown">digits</span></div></div>
<div class="giant"><b>statrs</b>the maths</div>
</div>
<div class="stack side">
<div class="giant"><b>scipy</b>the vocabulary, and the ruler</div>
</div>
</div>

<table class="frame spine-frame">
<thead><tr><th>sensor</th><th>reading</th><th>mu</th><th>sigma</th><th>z</th><th>upper_tail</th><th>simulated</th><th>log_tail</th></tr></thead>
<tbody>
<tr><td>temp-a</td><td>9.8</td><td>10.0</td><td>0.5</td><td>-0.4</td><td>0.655422</td><td class="todo">TODO</td><td>-0.422476</td></tr>
<tr><td>temp-a</td><td>10.4</td><td>10.0</td><td>0.5</td><td>0.8</td><td>0.211855</td><td class="todo">TODO</td><td>-1.551851</td></tr>
<tr><td>temp-a</td><td>40.0</td><td>10.0</td><td>0.5</td><td>60.0</td><td>0.0</td><td class="todo">TODO</td><td>-1805.013561</td></tr>
<tr><td>temp-b</td><td>99.4</td><td>100.0</td><td>2.0</td><td>-0.3</td><td>0.617911</td><td class="todo">TODO</td><td>-0.48141</td></tr>
<tr><td>temp-b</td><td>101.2</td><td>100.0</td><td>2.0</td><td>0.6</td><td>0.274253</td><td class="todo">TODO</td><td>-1.293704</td></tr>
<tr><td>temp-b</td><td>250.0</td><td>100.0</td><td>2.0</td><td>75.0</td><td>0.0</td><td class="todo">TODO</td><td>-2817.736604</td></tr>
</tbody>
</table>

<p class="footnote">&lt;TODO: simulated column, as on the order slide&gt;</p>

<!--
Back to the stack. I wrote none of Polars' engine and none of statrs' maths, and scipy gave me the vocabulary and the ruler. polars-stats, the thin strip in the middle, carries every pin: order, length, every row, digits. Every bug that mattered lived there, in a promise nobody had made me.
-->

---
layout: statement
---

# Borrow the engine, borrow the maths, and enforce what neither promises you inside the plugin, before it becomes a bug

<!--
So if you take one thing home: borrow the engine, borrow the maths, and enforce what neither of them promises you inside your own code, at the first line of the function, before it becomes a bug. And when the bug is in statrs or Polars, report it upstream with a reproducer.
-->

---

# Thank you to scipy, Polars and statrs

<div class="pair even credits">
<div>
<p>and to the people behind pyo3-polars, rand, mpmath and hypothesis</p>
<p>statrs fixes for these reports: day01 and teddytennant, merged by YeungOnion</p>
<p>polars-stats: sidsri14 (new distributions), camriddell (docs)</p>
</div>
<div>
<p class="install"><code>pip install polars-stats</code></p>
<p>Docs: <a href="https://fbruzzesi.github.io/polars-stats/">fbruzzesi.github.io/polars-stats</a></p>
<p>Code: <a href="https://github.com/FBruzzesi/polars-stats">github.com/FBruzzesi/polars-stats</a></p>
<p>Plugins: <a href="https://docs.pola.rs/user-guide/plugins/">docs.pola.rs/user-guide/plugins</a> · statrs: <a href="https://docs.rs/statrs">docs.rs/statrs</a></p>
</div>
</div>

<!--
Thank you to scipy, to Polars and to statrs, and to the people behind pyo3-polars, rand, mpmath and hypothesis. Thank you to the statrs contributors who turned my reports into fixes, to sidsri14, who contributed two distributions to polars-stats, and to camriddell, who fixed its docs. What I vouch for is the behaviour, and the tests that pin it. It's `pip install polars-stats`, and the docs are on the slide. Thank you.
-->
