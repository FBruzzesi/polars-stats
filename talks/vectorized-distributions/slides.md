---
theme: none
title: Vectorized statistical distributions, built on the Polars engine
info: 25 minutes plus 5 of Q&A
favicon: /logo.svg
author: Francesco Bruzzesi
colorSchema: light
canvasWidth: 1280
aspectRatio: 16/9
codeCopy: false
transition: fade
mdc: true
fonts:
  sans: Inter
  mono: JetBrains Mono
  provider: none
layout: default
class: cover
event: true
---

# Vectorized statistical distributions,<br>built on the Polars engine

<p class="tagline"><code>scipy.stats</code>-style distributions as native Polars expressions</p>

<p class="byline">Francesco Bruzzesi</p>

<img class="cover-bell" src="/cover-bell.svg" alt="">

---
clicks: 1
---

# Two of these six readings are faults. Which?

<div class="cols wide-left middle bells">
<Frame :cols="['sensor', 'reading', 'mu', 'sigma', 'z']" :ghost="$clicks >= 1 ? [] : ['z']" :flag="$clicks >= 1 ? 'reading' : undefined" fresh="z" />
<img v-click="1" src="/bells.svg" alt="One bell curve per sensor, the upper tail beyond a healthy reading shaded, each fault far off the chart">
<p v-click="1" class="caption">z: how far · the tail: how surprising</p>
</div>

<div v-click="1" class="caption">

$z = \dfrac{\texttt{reading} - \mu}{\sigma}$: how many standard deviations a reading sits from its own sensor's mean

</div>

<!--
Six temperature readings from two sensors, and each sensor has its own baseline: a mean, mu, and a standard deviation, sigma. Sensor A sits around 10 degrees, give or take half a degree; sensor B around 100, give or take 2. Two of these readings are faults. Which ones? [pause]

[click] Forty on A, two hundred and fifty on B. You just did a z-score in your head: how many standard deviations a reading sits from its own sensor's mean.

What we want to know is how surprising or likely is a reading at least this high? That's the upper tail of the distribution, its survival function, say "one in a thousand", instead of a threshold in degrees per sensor.

Alright folks, I'm Francesco, I'm **not** a Rust expert, and a good part of the Rust in polars-stats, the library this talk is about, was written with AI assistance. That turned out not to matter much: two big libraries hand you the engine and the maths for free. This talk is about four things neither of them promises, which is where all my bugs lived.
-->

---
class: section
eyebrow: 01 / Two libraries
---

# Polars has the data.<br>SciPy has the maths.<br>Why not use both?

<p class="subtitle">What each one gives you, and the glue most of us already write</p>

<!--
If you work in Python, two common options are Polars for the data handling and SciPy for statistical analysis. Let's quickly see both, and the glue between them. If you know them, this takes two minutes. If you don't, this is all you need.
-->

---
eyebrow: 01 · Polars
---

# Polars: describe the whole query, then the engine runs it

<div class="cols wide-left small-code">
<div class="window" data-file="query.py" data-lang="Python">

````md magic-move {at:1}
```python
import polars as pl

readings = pl.scan_parquet("readings.parquet")
```
```python
import polars as pl

readings = pl.scan_parquet("readings.parquet")
scored = readings.with_columns(
    z=(pl.col("reading") - pl.col("mu")) / pl.col("sigma"),
)
```
```python
import polars as pl

readings = pl.scan_parquet("readings.parquet")
scored = readings.with_columns(
    z=(pl.col("reading") - pl.col("mu")) / pl.col("sigma"),
).filter(pl.col("sensor") == "temp-b")
```
```python
import polars as pl

readings = pl.scan_parquet("readings.parquet")
scored = readings.with_columns(
    z=(pl.col("reading") - pl.col("mu")) / pl.col("sigma"),
).filter(pl.col("sensor") == "temp-b")

scored.collect(engine="in-memory")  # only now it runs,
scored.collect(engine="streaming")  # on either engine
```
````

</div>
<ul class="points">
<li v-click="1"><b>Expressions</b>column arithmetic, compiled, on every core</li>
<li v-click="2"><b>Lazy</b>a plan first; the optimiser rewrites it before a row is read</li>
<li v-click="3"><b>Two engines</b><code>"in-memory"</code>, or <code>"streaming"</code> in small batches called morsels</li>
</ul>
</div>

<p v-click="4" class="callout">No probability distributions: a query engine, not a statistics package</p>

<p class="footnote">pola.rs · a dataframe engine written in Rust, driven mostly from Python</p>

<!--
My data lives in Polars, a dataframe engine written in Rust that most people drive from Python. The word to remember is lazy.

You start from a file. [click] You describe new columns with expressions: reading minus mu, divided by sigma. That's compiled arithmetic, on every core.

[click] Then add a filter. Nothing has run yet. Polars holds a plan, and its optimizer rewrites that plan before it reads a single row.

[click] Only `collect` runs it, and you pick the engine: in memory, or streaming, in small batches it calls morsels, so the data never has to fit in memory. Same query, same answer. Remember morsels, they come back.

[click] What Polars doesn't have is statistics: no probability distributions. And that's fair enough: it's a query engine, not a statistics package.
-->

---
eyebrow: 01 · SciPy
---

# `scipy.stats`: a hundred distributions, one vocabulary

<table class="plain">
<thead><tr><th>pdf</th><th>cdf</th><th>sf</th><th>ppf · isf</th><th>logpdf · logsf</th><th>rvs</th></tr></thead>
<tbody><tr><td>density</td><td>probability below a value</td><td>probability above it</td><td>probability back to a value</td><td>the same, in logs</td><td>draw samples</td></tr></tbody>
</table>

<div class="window small-code" data-file="score.py" data-lang="Python">

````md magic-move {at:1}
```python
from scipy import stats

stats.norm(loc=10.0, scale=0.5).sf(10.4)
# 0.211855…
```
```python
import numpy as np
from scipy import stats

mu = np.array([10.0, 10.0, 100.0])
sigma = np.array([0.5, 0.5, 2.0])
x = np.array([9.8, 10.4, 101.2])

stats.norm(loc=mu, scale=sigma).sf(x)
# array([0.65542174, 0.2118554 , 0.27425312])
```
```python
import numpy as np
from scipy import stats

mu = np.array([10.0, 10.0, 100.0])
x = np.array([9.8, 10.4, 101.2])

stats.norm(loc=mu, scale=0.5).sf(x)  # one scale, stretched to every element
# array([0.65542174, 0.2118554 , 0.00819754])
```
````

</div>

<p v-click="2" class="callout">Broadcasting: arrays line up element by element, and a single value is stretched to their length</p>

<p class="footnote">scipy.org · Python's scientific library, since 2001</p>

<!--
Statistics in Python means SciPy, which is around since 2001. Its `stats` module has more than a hundred distributions, and they all speak the vocabulary on the slide. `sf`, the survival function, is the one we want from the first slide: the probability of a reading at least this high.

[click] And SciPy already does the per-row part. Give it an array of means, an array of sigmas, an array of readings, and it scores every element against its own distribution, in compiled code, with no Python loop.

[click] Give it one number for the scale and it stretches that number to the length of the arrays. That's called -_broadcasting_, and it comes from NumPy, Python's array library.
-->

---

# The round trip works, and the lazy query ends at `collect`

<div class="cols wide-left small-code">
<div class="window" data-file="round_trip.py" data-lang="Python">

```python {all|4|5,8-11|all}{at:1}
import polars as pl
from scipy import stats

df = pl.scan_parquet("readings.parquet").collect()
mu, sigma, x = df.select("mu", "sigma", "reading").to_numpy().T
baseline = stats.norm(loc=mu, scale=sigma)

df = df.with_columns(
    upper_tail=pl.Series(baseline.sf(x)),
    simulated=pl.Series(baseline.rvs(random_state=42)),
)
```

</div>
<div class="cards">
<div v-click="1" class="card"><b>The query ends</b><code>collect</code> runs everything above it; the optimiser never sees what comes after</div>
<div v-click="2" class="card"><b>Alignment is yours</b>bare arrays out, bare arrays in, through every later join and filter</div>
<div v-click="3" class="card bad"><b>NumPy has no null</b>missing data and invalid parameters both come back as <code>NaN</code></div>
</div>
</div>

<p class="footnote">docs › Explanation › Why polars-stats › What SciPy already does well</p>

<!--
So the first attempt is this one: take the columns out of Polars as NumPy arrays, call SciPy, put the results back. That's a naive approach, but it works. It costs three things:

[click] First, the query. To leave a lazy query you have to run it, so everything after collect is a second query, and the optimizer never sees the two together.

[click] Second, alignment. What comes back is a bare array, and keeping it on the right rows through the next join or filter is your job.

[click] Third, NumPy has no null, so missing data and invalid parameters both come back as NaN. Let's move to a quick quiz now.
-->

---
clicks: 1
---

# What does SciPy return for a normal distribution with standard deviation −1?

<div class="window" data-file="quiz.py" data-lang="Python">

```python
from scipy import stats

stats.norm(loc=0, scale=-1).sf(1.0)
```

</div>

<div v-click="1" class="reveal alarm">nan</div>

<p v-click="1" class="caption center">no exception, no warning · SciPy 1.18.1</p>

<!--
A normal distribution with a standard deviation of minus one. That isn't a distribution. What does SciPy give you for its tail? [pause]

[click] nan. No exception, no warning. That's a modeling error, travelling down your pipeline dressed up as missing data.
-->

---
class: section
eyebrow: 02 / The plugin
---

# Can a distribution be a Polars expression instead?

<p class="subtitle">Keep the query whole, and keep every answer on its own row</p>

<!--
So: could the distribution live inside the Polars query instead?
-->

---
eyebrow: 02 · The plugin
---

# Polars has `erfc` now. It still has no distributions

<div class="formula">

$$
\texttt{upper\_tail} = \tfrac{1}{2}\,\operatorname{erfc}\!\left(\frac{z}{\sqrt{2}}\right)
$$

<span class="pill ok">Polars 2: <code>erfc</code> built in</span>
</div>

<div class="cards">
<div v-click="1" class="card bad"><b>Polars' own <code>erfc</code></b>one tail of one distribution: nothing for Gamma's or Beta's incomplete functions, inverse cdfs, log tails or seeded draws</div>
<div v-click="2" class="card bad"><b>Back to Python: <code>map_elements</code></b>Polars' escape hatch runs a Python function, SciPy inside, once per row: through the interpreter, on one thread</div>
<div v-click="3" class="card move"><b>An expression plugin</b>a compiled Rust function that takes columns and returns a column; Polars calls it as one of its own expressions. The maths is still yours to write</div>
</div>

<!--
Polars already has the first half, the z-score. Turning a z-score into a tail probability takes a special function, the error function, and recent Polars ships with it: for the Normal, the tail is one line. But what would you reach for when the distribution isn't Normal? [pause]

[click] Polars' own erfc covers one tail of one distribution. A Gamma or a Beta needs incomplete gamma and beta functions, a quantile needs an inverse, the far tail needs logarithms, and simulation needs seeded draws. Polars has none of those, and it shouldn't: **it's a query engine**.

[click] So you go back to Python. Polars has an escape hatch for that, map_elements: it calls a Python function, with SciPy inside, once per row. It works, through the interpreter, on one thread.

[click] The solution is to use the door the Polars team builds and documents: an expression plugin. You write a Rust function that takes columns and returns a column, and Polars calls it as one of its own expressions. That solves the engine side. The maths is still under our responsibility.

And I was not going to write all of it myself.
-->

---
eyebrow: 02 · statrs
---

# statrs: the closest thing Rust has to `scipy.stats`

<div class="cols wide-left">
<div class="window" data-file="src/main.rs" data-lang="Rust">

```rust {all|5|6-7|8-9|all}{at:1}
// Cargo.toml: statrs = "0.19"
use statrs::distribution::{ContinuousCDF, Normal};

fn main() {
    let baseline = Normal::new(10.0, 0.5).unwrap();
    println!("{}", baseline.sf(10.4));
    // 0.211855…
    println!("{:?}", Normal::new(0.0, -1.0));
    // Err(StandardDeviationInvalid)
}
```

</div>
<ul class="points">
<li><b>Per distribution</b><code>pdf</code>, <code>cdf</code>, <code>sf</code>, <code>inverse_cdf</code>, <code>ln_pdf</code>, sampling</li>
<li><b>Underneath</b><code>erf</code>, gamma and beta functions</li>
<li v-click="4"><b>One distribution, one value</b>it knows nothing about columns</li>
</ul>
</div>

<p class="footnote">docs.rs/statrs · a Rust crate (Rust's word for a library), maintained by volunteers</p>

<!--
So the maths comes from a second library, on the Rust side: `statrs`. It's a crate, which is Rust's word for a library, it's the closest thing Rust has to scipy.stats, and it's maintained by volunteers.

[click] Build a Normal distribution from a mean and a standard deviation.

[click] Ask for its survival function: the same answer as SciPy.

[click] Ask for a Normal with sigma minus one and you get an error, not a nan: statrs checks its parameters when you build the distribution. Underneath sit erf, gamma and beta.

[click] What statrs doesn't do is columns, vectorized operations. A statrs distribution is one distribution, evaluated at one value. Someone has to walk the rows.
-->

---
eyebrow: 02 · The plugin
---

# The SciPy call, as a Polars expression

<div class="cols middle">
<div class="window" data-file="score.py" data-lang="Python">

````md magic-move {at:1}
```python
from scipy import stats

baseline = stats.norm(loc=mu, scale=sigma)
upper_tail = baseline.sf(x)
```
```python
import polars as pl
import polars_stats as ps

baseline = ps.Normal(mu="mu", sigma="sigma")
scored = readings.with_columns(
    upper_tail=baseline.sf("reading"),
)
```
````

</div>
<div v-click="2"><Frame :cols="['sensor', 'reading', 'mu', 'sigma', 'upper_tail']" fresh="upper_tail" /></div>
</div>

<!--
That someone is `polars-stats`. Here's the SciPy call. [click] And here's the same call in polars-stats. sf is still sf, but the parameters are column names and the result is a Polars expression, not an array.

[click] It plugs into `with_columns`, it stays inside the lazy query, and every row is scored against its own sensor. If you come from R, this is `pnorm` inside `mutate`, which R users have always had.
-->

---
eyebrow: 02 · The plugin
---

# Python plans, Rust walks the rows, statrs computes

<div class="cols anatomy small-code">
<div class="window" data-file="polars_stats/distributions/_base.py" data-lang="Python">

```python {all|3-6|0|0|7}{at:1}
import polars as pl

pl.plugins.register_plugin_function(
    plugin_path=LIB,
    function_name="normal_sf",
    args=[x, mu, sigma],
    is_elementwise=True,
)
```

</div>
<div class="window" data-file="src/distributions/normal.rs" data-lang="Rust">

```rust {all|0|5-8|11|0}{at:1}
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use statrs::distribution::{ContinuousCDF, Normal};

#[polars_expr(output_type=Float64)]
fn normal_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, sf_value)
}

fn sf_value(dist: &Normal, v: f64) -> Option<f64> {
    Some(dist.sf(v)) // statrs
}
```

</div>
</div>

<p v-click="4" class="callout"><code>is_elementwise=True</code> is a promise: row i of the output depends only on row i of the inputs</p>

<p class="footnote">simplified from the source · docs › Explanation › Architecture › Three layers</p>

<!--
So here's the whole anatomy, in three layers.

[click] Python: the method returns an expression and computes nothing. Underneath, it registers a call to a Rust function, by name.

[click] Rust: that function gets the columns, walks the rows and builds one statrs Normal per row.

[click] And this one line, dist dot sf, is statrs.

[click] One flag in that registration matters more than the rest: `is_elementwise=True`. It's a promise to Polars: "The i-th output row depends only on row i of the inputs". Watch what the optimiser does with it.
-->

---
eyebrow: 02 · The plugin
---

# One flag lets the optimiser move the filter

<div class="cols plan small-code">
<div class="window" data-file="pipeline.py" data-lang="Python">

```python
import polars as pl
import polars_stats as ps

baseline = ps.Normal(mu="mu", sigma="sigma")
scored = (
    pl.scan_parquet("readings.parquet")
    .with_columns(upper_tail=baseline.sf("reading"))
    .filter(pl.col("sensor") == "temp-b")
    .filter(pl.col("upper_tail") < 1e-3)
)
print(scored.explain())
```

</div>
<div v-click="1" class="plan-col">
<p class="promise" :class="$clicks >= 2 ? 'on' : 'off'"><code>is_elementwise={{ $clicks >= 2 ? 'True' : 'False' }}</code></p>
<div class="window" data-file="scored.explain()" data-lang="plan">

````md magic-move {at:2}
```text
FILTER (col("sensor") == "temp-b") &
       (col("upper_tail") < 0.001)
FROM
  WITH_COLUMNS:
  [… normal_sf(…) …]
    Parquet SCAN [readings.parquet]
    PROJECT */4 COLUMNS
```
```text
FILTER col("upper_tail") < 0.001
FROM
  WITH_COLUMNS:
  [… normal_sf(…) …]
    Parquet SCAN [readings.parquet]
    PROJECT */4 COLUMNS
    SELECTION: col("sensor") == "temp-b"
```
````

</div>
</div>
</div>

<p class="footnote">Polars 2.1.0 · a 1,000,000-row Parquet file of the two sensors · plugin node shortened · read the plan from the bottom up</p>

<!--
We have a pipeline that:
1. Scans a Parquet file, a common columnar file format, holding a million readings
2. Adds the tail probability
3. Keeps sensor B
4. Keeps the readings rarer than one in a thousand.

[click] First without the promise: `is_elementwise=False`. Here's the plan Polars builds, read from the bottom up. The scan reads every row, every row gets scored, and only then do both filters run, at the top. If my function could look at its neighbours (since it's not elementwise), removing rows first might change the answers, so Polars won't risk it.

[click] Now flip the flag to `True`. Same query. [pause] The filter on sensor B, which I wrote after the scoring, has moved down into the file reader. Half the rows are never decoded, so my function never sees them. That's predicate pushdown, and I wrote none of it. The promise we make to polars buys the optimisation.
-->

---

# Polars runs it, statrs computes it. What is left for me to write?

<Stack :pins="0" />

<!--
So here's the picture: The polars engine on top, the statrs math at the bottom, SciPy on the side with its vocabulary and acting as a ruler to test against.

In between, a thin strip of code that I actually wrote: `polars-stats`.

What goes in the strip? Polars hands a plugin columns. statrs takes one value. In between sit four promises that neither of them makes, and each one cost me a bug. Let's fill them in.
-->

---
class: section
eyebrow: 03 / Reproducibility
---

# Same seed, same column?

<p class="subtitle">The second task: simulated readings, identical on every run</p>

<!--
The first one arrived with the other task: simulating.
-->

---
eyebrow: 03 · Reproducibility
---

# One generator walking the rows depends on where Polars cuts them

<div class="cols small-code">
<div class="window" data-file="src/expressions.rs, first version" data-lang="Rust">

```rust
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use statrs::distribution::Bernoulli;

let mut rng = ChaCha20Rng::seed_from_u64(seed);
for i in 0..n {
    let dist = Bernoulli::new(p[i])?;
    values.push(dist.sample(&mut rng));
}
```

</div>
<div class="pieces" :class="{ restart: $clicks >= 2 }">
<div class="lane whole"><span>the column</span><div class="bar"><span class="piece"></span></div></div>
<div class="lane"><span>chunks</span><div class="bar"><span class="piece"></span><span class="piece"></span><span class="piece"></span></div></div>
<div class="lane"><span>threads</span><div class="bar"><span class="piece"></span><span class="piece"></span><span class="piece"></span><span class="piece"></span></div></div>
<div class="lane"><span>morsels</span><div class="bar"><span class="piece"></span><span class="piece"></span><span class="piece"></span><span class="piece"></span><span class="piece"></span><span class="piece"></span></div></div>
<p v-click="2" class="caption">↺ 42: the generator restarts from the seed in every piece</p>
</div>
</div>

<p v-click="1" class="predict">Same frame, same seed: in-memory engine against streaming engine. Same numbers?</p>

<p v-click="2" class="answer"><b class="alarm">No.</b> The seed reproduced the chunk layout, not the query.</p>

<!--
The first version of polars-stats had one method: `sample`. The idea was simple: seed one random number generator, walk the rows in order, draw one value per row.

Because Polars doesn't hand a plugin the column. It hands it pieces: chunks in memory, splits across threads, morsels when streaming, one group at a time under over. **The function runs once per piece**.

[click] So, same frame, same seed, in-memory engine against streaming engine. Do we get the same numbers? [pause]

[click] No. Every call restarts the generator from seed 42, and where the pieces start depends on the engine, not on your data. The seed was reproducing the chunk layout, not the query.
-->

---
eyebrow: 03 · Reproducibility
---

# Every row draws from its own generator, keyed on `(seed, row index)`

<div class="cols wide-left middle">
<div class="window" data-file="src/rng.rs, simplified" data-lang="Rust">

````md magic-move {at:1}
```rust
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use statrs::distribution::Bernoulli;

let mut rng = ChaCha20Rng::seed_from_u64(seed);
for i in 0..n {
    let dist = Bernoulli::new(p[i])?;
    // every draw moves the one shared stream forward
    values.push(dist.sample(&mut rng));
}
```
```rust
use rand_pcg::Pcg64Mcg;
use statrs::distribution::Bernoulli;

for i in 0..n {
    // a fresh generator per row, from (seed, row index)
    let mut rng: Pcg64Mcg = row_rng(seed, index[i]);
    let dist = Bernoulli::new(p[i])?;
    values.push(dist.sample(&mut rng));
}
```
````

</div>
<table class="plain dice">
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
</div>

<p v-click="2" class="callout">The row index is one more input column, so Polars carries it through chunks, morsels and groups</p>

<!--
**The fix** is to stop thinking of one stream walking down the rows. [click] Rather, give every row its own generator, built from two numbers: the seed and the row's index. Row 3 draws from the stream for 42 and 3, whichever piece, thread or engine it lands in. Cut the pieces anywhere: every row keeps its die.

[click] And the row index is just another input column, so Polars carries it through chunks, morsels and groups for me.
-->

---
eyebrow: 03 · Reproducibility
---

# A seeded column repeats across runs and engines

<div class="cols wide-left small-code">
<div class="window" data-file="simulate.py" data-lang="Python">

```python
import polars as pl
import polars_stats as ps

baseline = ps.Normal(mu="mu", sigma="sigma")
sim = readings.with_columns(simulated=baseline.sample(seed=42))
a = sim.collect(engine="in-memory")
b = sim.collect(engine="streaming")
a.equals(b)  # True
```

</div>
<div v-click="1"><Frame :cols="['sensor', 'mu', 'sigma', 'simulated']" fresh="simulated" /></div>
</div>

<p v-click="2" class="caption">The bill: a generator per row has to be cheap to build. <s><code>ChaCha20</code></s> per row was 10 to 20× slower; <code>Pcg64Mcg</code> took 1M rows × 100 draws from 5.9 s to 0.29 s</p>

<Stack v-click="3" mini :pins="1" />

<p class="footnote">docs › Explanation › Design notes › Sampling derives a fresh per-row RNG</p>

<!--
Today: same seed, same column, on both engines, and CI runs every test once per engine.

[click] Here's our frame with its simulated column: one plausible reading per row, from that row's own sensor, the same on every run.

[click] The fix came with a bill. My first generator was a cryptographic one, and building one per row made sampling ten to twenty times slower. A small statistical generator, Pcg64Mcg, costs a handful of integer operations to build: a million rows times a hundred draws went from 5.9 seconds to 0.29.

[click] First pin: **reproducibility**.
-->

---
class: section
eyebrow: 04 / Broadcast
---

# Does every input arrive at full length?

<p class="subtitle">NumPy stretches a single value. What reaches a plugin?</p>

<!--
The plugin gets pieces. Next question: do all its inputs arrive at full length?
-->

---
eyebrow: 04 · Broadcast
---

# NumPy broadcasts. Polars does not broadcast into a plugin

<div class="cols">
<div>
<h3>NumPy and SciPy</h3>
<div class="window" data-file="SciPy" data-lang="Python">

```python
stats.norm(loc=mu, scale=0.5).sf(x)
# scale: 1 value, stretched to len(x)
```

</div>
</div>
<div v-click="1">
<h3>A Polars plugin</h3>
<div class="window" data-file="the plugin's inputs" data-lang="Rust">

```rust
fn normal_sf(inputs: &[Series])
// lengths: 1_000_000, 1_000_000, 1
```

</div>
<p class="caption">no broadcasting into a plugin, and the helpers that walk several columns stop at the shortest</p>
</div>
</div>

<div v-click="2" class="trio">
<div class="card ok"><b><code>sigma=0.5</code></b>a Python number: Python can see it and pad it</div>
<div class="card bad"><b><code>sigma=pl.lit(0.5)</code></b>one row long, and only the engine knows it</div>
<div class="card bad"><b><code>sigma=r.std()</code></b>one row long, and only the engine knows it</div>
</div>

<!--
Remember broadcasting: give SciPy one sigma and it stretches it to the length of the readings. Polars does the same inside its own arithmetic.

[click] But not into a plugin. My function gets its inputs as they are: a million readings, a million means, and one sigma. And the Rust helpers that walk several columns together stop at the shortest one.

[click] The first version handled the case I thought of: a plain Python number, padded in Python. But you can also write `pl.lit(0.5)`, a constant wrapped as an expression, or a column's standard deviation. Each is one row long, and only the running engine knows it.
-->

---
eyebrow: 04 · Broadcast
---

# Expected: 1,000,000 distinct values. Got?

<div class="window small-code" data-file="broadcast.py, polars-stats 0.0.1 from PyPI" data-lang="Python">

```python
import numpy as np
import polars as pl
import polars_stats as ps

frame = pl.LazyFrame({"x": np.linspace(-3, 3, 1_000_000)})
cdf = ps.Normal(mu=0.0, sigma=pl.lit(0.5)).cdf("x")
frame.select(cdf).collect(engine="streaming").n_unique()
```

</div>

<div class="trio">
<div v-click="1"><div class="stat mid alarm">1</div><p class="stat-label">in-memory engine</p></div>
<div v-click="1"><div class="stat mid alarm">10</div><p class="stat-label">streaming, one per morsel</p></div>
<div v-click="2"><div class="stat word">no error</div><p class="stat-label">1,000,000 rows on both</p></div>
</div>

<p class="footnote">polars-stats 0.0.1 and Polars 1.43.2, from PyPI in a clean environment · current release: 1,000,000</p>

<!--
Version 0.0.1, straight from PyPI, Python's package index. A million rows, a Normal with sigma equal to `pl.lit(0.5)`, scored with `cdf`. The right answer has a million distinct values. How many did we get? [pause]

[click] In-memory: one. Row zero's answer, a million times. Streaming: ten, one per morsel.

[click] The right height, no error, no warning. That one I published.
-->

---
eyebrow: 04 · Broadcast
---

# Align inputs by their length at run time, in Rust, first

<div class="cols small-code">
<div class="window" :data-file="$clicks >= 1 ? 'src/distributions/mod.rs, align_inputs, abridged' : 'the first version, Python'" :data-lang="$clicks >= 1 ? 'Rust' : 'Python'">

````md magic-move {at:1}
```python
if isinstance(sigma, pl.Expr):
    self._sigma = sigma  # passed through
elif isinstance(sigma, float):
    self._sigma = pl.repeat(sigma, n=pl.len())
```
```rust
for input in inputs {
    if input.len() == 1 {
        needs_broadcast = true;  // stretch it
    } else if let Some(anchor) = anchor {
        polars_ensure!(
            input.len() == anchor.len(),
            ShapeMismatch: "…"
        );
    } else {
        anchor = Some(input);
    }
}
```
````

</div>
<div v-click="2">
<div class="window" data-file="per_sensor.py" data-lang="Python">

```python
import polars as pl
import polars_stats as ps

r = pl.col("reading")
per_sensor = ps.Normal(mu=r.mean(), sigma=r.std())
history.with_columns(
    upper_tail=per_sensor.sf(r).over("sensor"),
)
```

</div>
<p class="caption"><code>r.mean()</code> is one value per group: the plugin stretches it, Polars does the grouping</p>
</div>
</div>

<Stack v-click="3" mini :pins="2" />

<p class="footnote">a breaking release, 0.0.2 · docs › Explanation › Architecture › Column-valued parameters</p>

<!--
The fix: [click] stop judging by the kind of expression, in Python, and judge by length, at run time, in Rust. The first thing every plugin does now is this loop: an input of length one gets stretched to the others' length, and two real lengths that disagree raise a shape error. It changed what some calls return, so it shipped as a breaking release.

[click] And it buys this line: estimate each sensor's baseline from its own history and score against it, in one expression, inside an over. The mean and the standard deviation are one value per sensor, length one. The plugin stretches them, Polars does the grouping.

[click] Second pin: **broadcast**.
-->

---
class: section
eyebrow: 05 / Parameter validation
---

# Does every row reach the parameter check?

<p class="subtitle">SciPy answers an impossible parameter with NaN. We wanted an error</p>

<!--
Every input arrives at full length. Now back to SciPy's nan: can we raise instead?
-->

---
eyebrow: 05 · Parameter validation
---

# An invalid parameter raises. A null stays null

<div class="cols small-code">
<div class="window" data-file="SciPy" data-lang="Python">

```python
from scipy import stats

stats.norm(loc=0, scale=-1).sf(1.0)
# nan
```

</div>
<div v-click="1" class="window" data-file="polars-stats" data-lang="Python">

```python
import polars as pl
import polars_stats as ps

pl.select(ps.Normal(mu=0.0, sigma=-1.0).sf(1.0))
# ComputeError: sigma must be finite
#   and strictly positive, got -1
```

</div>
</div>

<table v-click="2" class="plain">
<thead><tr><th>on a row</th><th>result</th></tr></thead>
<tbody>
<tr><td>a null value or a null parameter</td><td><code>null</code></td></tr>
<tr><td>a <code>NaN</code> value</td><td><code>NaN</code>, as in SciPy</td></tr>
<tr><td>an invalid parameter, on any row</td><td><code>ComputeError</code>: the whole query fails</td></tr>
</tbody>
</table>

<p v-click="3" class="callout">A Polars expression cannot raise on a row, so the check is a plugin, and it has to see every row</p>

<!--
Here's SciPy's answer again: nan. [click] The same call in polars-stats: a ComputeError that names the parameter and the bad value.

[click] The contract, straight from the docs: a null in gives a null out, a NaN value gives NaN, as in SciPy, and an invalid parameter on any row fails the whole query.

[click] There's a catch. A Polars expression cannot raise on a row. So the check has to be a plugin, and it has to see every row.
-->

---
eyebrow: 05 · Parameter validation
---

# Every parameter is checked in Rust, before the maths

<div class="window" data-file="normal_sf, illustrative" data-lang="Rust">

```rust {all|2|3|5-10}{at:1}
fn normal_sf(inputs: &[Series]) -> PolarsResult<Series> {
    let [x, mu, sigma] = align_inputs(inputs)?;
    check_sigma(&sigma)?;
    let mut out = Vec::new();
    for i in 0..x.len() {
        out.push(match (x[i], mu[i], sigma[i]) {
            (Some(x), Some(mu), Some(s)) => Some(Normal::new(mu, s)?.sf(x)),
            _ => None,
        });
    }
    Ok(out.into())
}
```

</div>

<Stack v-click="4" mini :pins="3" />

<p class="footnote">illustrative, not the source · docs › Explanation › Design notes › Invalid parameters raise, they never silently null</p>

<!--
So every method that takes a value is a Rust plugin, and inside it the order is fixed. Here's the shape of it for the Normal's sf. [click] First align the inputs: that's broadcast. [click] Then check the whole sigma column, every row, before any distribution is built. [click] Only then the maths, row by row: a null anywhere on the row gives a null, and a NaN reading goes through statrs and comes out NaN.

[click] Third pin: **parameter validation**.
-->

---
class: section
eyebrow: 06 / Accuracy
---

# Is the number on each row right?

<p class="subtitle">Every method agrees with SciPy. Is that enough?</p>

<!--
Every row is checked now. Is the number on each row right?
-->

---
eyebrow: 06 · Accuracy
---

# In the far tail, agreeing with SciPy proves nothing

<ul class="points">
<li><b>Parity</b>every method of every distribution, against SciPy, on the same inputs</li>
<li><b>Property tests</b>Hypothesis invents thousands of inputs and checks what must always hold: a cdf stays in [0, 1] and never goes down</li>
</ul>

<table v-click="1" class="plain">
<thead><tr><th>the 40.0 reading's tail: <code>Normal(10, 0.5).sf(40.0)</code></th><th>answer</th></tr></thead>
<tbody>
<tr><td>SciPy, float64</td><td><code>0.0</code></td></tr>
<tr><td>polars-stats, float64</td><td><code>0.0</code></td></tr>
<tr v-click="2"><td>mpmath, 50 significant digits</td><td><code>1.23757e-784</code></td></tr>
</tbody>
</table>

<p v-click="2" class="caption">a float64 holds nothing below about 5e-324, so both libraries round to 0.0, and two zeros always agree</p>

<!--
Every method of every distribution is tested against SciPy on the same inputs, plus property tests: a tool called Hypothesis invents thousands of inputs and checks what must always hold, like a cdf staying between zero and one and never going down. All green.

[click] Here's our forty-degree fault. SciPy says its tail is zero, and so does polars-stats. They agree perfectly, and both are wrong: a 64-bit float can't hold a number that small, so both round it to zero. That's a float64 limit, not a SciPy one, and agreeing with SciPy can't tell you who's right there.

[click] So I brought a longer ruler: mpmath, a Python library that computes with as many digits as you ask for. We do 50.
The true tail is about ten to the minus 784.
-->

---
eyebrow: 06 · Accuracy
---

# A coin that comes up 1 with probability `1e-300`

<div class="window" data-file="bernoulli.py" data-lang="Python">

```python
import polars as pl
import polars_stats as ps

pl.select(ps.Bernoulli(p=1e-300).sf(0.5))  # P(value > 0.5)
```

</div>

<div class="derive">
<div v-click="1"><span>should be</span><code>1e-300</code></div>
<div v-click="2"><span>returned</span><code class="alarm">0.0</code></div>
<div v-click="3"><span>computed as</span><code>1 - cdf = 1 - (1 - p)</code></div>
<div v-click="4"><span>in float64</span><code>1 - 1e-300 == 1.0</code></div>
</div>

<p v-click="5" class="callout">Catastrophic cancellation: compute the quantity directly, and work in logarithms in the tails</p>

<!--
Most of the defects the oracle found were not in statrs: they were in closed forms I had written myself, one to three lines each. My favourite is a coin that comes up one with probability ten to the minus three hundred. What's the probability of a value above one half? [click] It's `p`, a perfectly representable number. [click] I returned zero. [click] Because I computed one minus the `cdf`, and the `cdf` is one minus `p`. [click] In 64-bit floats, one minus ten to the minus three hundred is exactly one: `p` was gone before the subtraction even happened.

[click] Subtracting two nearly equal numbers has a name: catastrophic cancellation. The fixes were small: compute what you want directly, not as a difference, and in the tails, work in logarithms.
-->

---
clicks: 1
eyebrow: 06 · Accuracy
---

# `log_sf` ranks the faults that `sf` rounds to zero

<div class="cols frame-left middle small-code">
<Frame :cols="['sensor', 'reading', 'z', 'upper_tail', 'log_tail']" :ghost="$clicks >= 1 ? [] : ['log_tail']" wrong="upper_tail" fresh="log_tail" />
<div class="window" data-file="score.py" data-lang="Python">

```python
import polars_stats as ps

baseline = ps.Normal(mu="mu", sigma="sigma")
upper_tail = baseline.sf("reading")
log_tail = baseline.log_sf("reading")
```

</div>
</div>

<p class="caption">true tails: about 1e-784 and 1e-1224, below anything a float64 holds · no algorithm can return them, the log can</p>

<!--
Back to our sensors. The true tails of the two faults are around ten to the minus 784 and ten to the minus 1224, far below anything a float64 can hold. `sf` says zero for both, so you can't rank them. And that zero is not a bug: it's the closest float64 to the truth. No algorithm can do better, so the fix is to ask for a different number.

[click] `log_sf` says minus 1805 and minus 2817, and the worse fault sorts first.
-->

---
clicks: 1
eyebrow: 06 · Accuracy
---

# `Beta(s, s).cdf(0.5)` is exactly `0.5`, for every `s`

<div class="window" data-file="src/main.rs, statrs 0.19.1" data-lang="Rust">

```rust
use statrs::distribution::{Beta, ContinuousCDF};

Beta::new(s, s)?.cdf(0.5) // 0.5 by symmetry
```

</div>

<div v-click="1" class="trio">
<div><div class="stat mid warn">0.4912</div><p class="stat-label">s = 1e6</p></div>
<div><div class="stat mid warn">0.2129</div><p class="stat-label">s = 1e7</p></div>
<div><div class="stat mid alarm">−1.147</div><p class="stat-label">s = 1e8: a probability below zero</p></div>
</div>

<!--
That was a limit of the format. The other kind is a limit of the algorithm, where a float64 could hold the answer and the code loses it. My coin was one; some were statrs'. A Beta distribution with two equal shapes is symmetric, so its cdf at one half is exactly one half, for every shape.

[click] At a million, statrs said 0.49. At ten million, 0.21. At a hundred million, minus 1.147. A probability below zero.
-->

---
eyebrow: 06 · Accuracy
---

# A reproducer upstream fixed it for every statrs user

<div class="shots">
<figure class="shot-issue">
<img src="/statrs-issue-434.png" alt="statrs issue 434: Beta::cdf loses accuracy above shapes of 1e4 and returns values outside [0, 1] by 1e8, with a table of statrs results against the exact 0.5">
<figcaption>statrs #434: the report, with its reproducer</figcaption>
</figure>
<figure v-click="1" class="shot-pr">
<img src="/statrs-pr-456.png" alt="statrs pull request 456, Stabilize large beta CDF, merged by the maintainer">
<figcaption>statrs #456: the fix, by another contributor</figcaption>
</figure>
</div>

<!--
I could have patched that inside polars-stats and moved on. That fixes it for one project. Instead it went upstream as an issue: the input, what statrs returns, and the exact answer, which here needs no oracle at all.

[click] Two and a half weeks later another contributor, day01, opened the fix, and the maintainer merged it. The root cause: a continued fraction capped at 140 iterations that returned its last value instead of an error. That report cost me an evening, and the fix reaches everyone who uses statrs, not just polars-stats.

It wasn't the only one. I found a few more issues in statrs along the way, and every time the maintainers were welcoming and supportive.
-->

---
eyebrow: 06 · Accuracy
---

# A defect has two acceptable outcomes, and no third

<blockquote class="quote">The algorithm gets fixed, or the caveat gets documented with a regime and a magnitude. A runtime warning is never the fix, because it cannot fire per row from inside the engine.</blockquote>

<Stack v-click="1" mini :pins="4" />

<p class="footnote">docs › Explanation › Numerical accuracy</p>

<!--
The rule in the docs: fix the algorithm, or document the limit, where it starts and how big it is. Every statrs issue still open is documented that way.

[click] Fourth pin: **accuracy**.

The most dangerous maths in polars-stats turned out to be the maths I wrote myself.
-->

---
class: section
eyebrow: 07 / Cost
---

# Was it worth it, and what does it cost?

<p class="subtitle">Before any speed number, everything wrong with the measurement</p>

<!--
Was it worth it? Before any speed number, here's everything wrong with the measurement.
-->

---
eyebrow: 07 · Cost
---

# Every number that follows comes from one laptop

<div class="stamp">Disclaimer</div>

<dl class="conditions">
<dt>Machine</dt><dd>Apple M5, 10 cores (4 performance, 6 efficiency), 24 GB, macOS 26.7.1, on AC power, other apps closed</dd>
<dt>Software</dt><dd>Python 3.14.10 · Polars 2.1.0 · SciPy 1.18.1 · NumPy 2.5.4 · polars-stats 0.1.0, release build</dd>
<dt>polars-stats</dt><dd>streaming engine, default thread pool (10 threads); the timed call includes building and dispatching the query</dd>
<dt>SciPy</dt><dd>the classic frozen API, <code>norm(loc=…, scale=…)</code> then the method; samplers get a PCG64 <code>Generator</code>; one thread</dd>
<dt>Regime</dt><dd>one parameter value per row, at 10k, 1M and 10M rows; plain numbers and one-value expressions are other code paths on both sides, not measured here</dd>
<dt>Not measured</dt><dd>SciPy's newer API (measured earlier on the Normal, column parameters: 1.3 to 1.9× faster than the frozen one); peak memory</dd>
<dt>samples</dt><dd>in the column regime SciPy draws <code>(draws, rows)</code> and transposes a view: optimistic for SciPy by about one pass over the output</dd>
<dt>Timing</dt><dd>median of up to 50 iterations within 10 s per cell (at least 3), after one warmup · ratio = SciPy / polars-stats, above 1 means polars-stats is faster</dd>
</dl>

<p class="footnote">run on 3 November 2026</p>

<!--
This is a local benchmark, on my laptop, so it might be wrong, like every benchmark. Everything you need to reproduce it is on this slide, and the slides are online.
-->

---
clicks: 4
eyebrow: 07 · Cost
---

# One value per row: large frames favour polars-stats

<div class="chart">
<img src="/bench-column.svg" alt="Dot plots of the SciPy to polars-stats time ratio per distribution, column regime, at 10k, 1M and 10M rows, for sample, samples and sf">
<img v-click="4" src="/bench-column-10k.svg" alt="Overlay: the 10k-row cells where SciPy is ahead">
<span class="dim sample" :class="{ on: $clicks === 2 || $clicks === 3 }"></span>
<span class="dim samples" :class="{ on: $clicks === 1 || $clicks === 3 }"></span>
<span class="dim sf" :class="{ on: $clicks === 1 || $clicks === 2 }"></span>
</div>

<p v-click="4" class="caption center">10k rows: SciPy ahead on <code>sample</code> for all but DiscreteUniform, up to 6.8×; polars-stats pays 0.3 to 0.45 ms a call</p>

<p class="footnote">column regime only · DiscreteUniform: SciPy's <code>randint</code> with per-row bounds runs <code>np.vectorize</code> · Bernoulli: NumPy <code>binomial(1, p)</code></p>

<!--
The per-row case, the reason polars-stats exists. First, how to read it. Each panel is a method, each line is a distribution. A dot is SciPy's time divided by polars-stats' time: on the dark vertical line they tie, and right of it polars-stats is faster. The axis is logarithmic, so each grid line is ten times more. Three dots per distribution, one per frame size: hollow for ten thousand rows, purple for a million, black for ten million.

[click] Simulation, one draw per row: at a million rows, 1.6 to 8 times faster than classic SciPy.

[click] Ten draws per row: 4.7 to 18 times.

[click] Scoring with sf: 2.3 to 8 times. The two starred lines at the bottom, discrete uniform and Bernoulli, go through slow paths in SciPy. Don't quote those.

[click] Now the hollow dots. At ten thousand rows SciPy wins every one-draw-per-row cell but one, by up to nearly seven times: polars-stats spends a third to almost half a millisecond planning and dispatching the query, whatever the distribution. Large frames amortise that; small frames pay it. And this is polars-stats plus Polars' thread pool, against classic SciPy, on this laptop: what you get when you use it.
-->

---
eyebrow: 07 · Cost
---

# Two days later, a friend rewrote polars-stats in Nim

<div class="cols wide-left middle">
<div class="cards">
<div v-click="1" class="card"><b>"Polars plugins, without the Rust"</b>the same 14 distributions as Nim plugins: about 780 lines, against about 6,300 of Rust. His verdict: "On par, as hoped, and no more than that."</div>
<div v-click="2" class="card move"><b>So I went back to work</b>checking a parameter column is now a tight loop the compiler can vectorise, and every answer is written in one pass, with a null mask built only when a null shows up</div>
</div>
<figure class="qr">
<img src="/qr-nanuq.svg" alt="QR code for pietroppeter.github.io/nanuq-more">
<figcaption>pietroppeter.github.io/nanuq-more</figcaption>
</figure>
</div>

<p class="footnote">line counts and verdict from the post · both changes ship in polars-stats 0.2.0</p>

<!--
Two days after that run, my friend Pietro, who loves the Nim programming language, published a post called "Polars plugins, without the Rust". [click] He rebuilt polars-stats in Nim: the same fourteen distributions, in about 780 lines, against about 6,300 of Rust. His verdict: on par, as hoped, and no more than that.

[click] I took it as a challenge. Two changes: checking a column of parameters became a tight loop the compiler can vectorise, and every answer is now written in a single pass, with a null mask built only when there's a null to record. Performance is never done, so I ran the same benchmark again.
-->

---
eyebrow: 07 · Cost
---

# polars-stats 0.2.0, same laptop, same benchmark

<div class="chart">
<img src="/after/bench-column.svg" alt="The same dot plots for polars-stats 0.2.0: SciPy to polars-stats time ratio per distribution, column regime, at 10k, 1M and 10M rows, for sample, samples and sf">
<figure v-click="1" class="shot-quote">
<img src="/nanuq-round-three.png" alt="From the post: Round three goes to Francesco, no contest. Nim's homework is now a precise list: a row loop with no per-row overhead, constant checks hoisted out of the loop, then the gamma quantile and the samplers. The ball is in Nim's court, and it is bouncing rather fast.">
<figcaption>nanuq-more, round three (updated 9 November)</figcaption>
</figure>
</div>

<p class="caption center">1M rows, vs 3 November: <code>sf</code> 1.3× further ahead (Uniform 3× → 12×), sampling 1.1 to 1.2×, Gamma flat</p>

<p class="footnote">run on 9 November 2026 · polars-stats 0.2.0 and macOS 26.7.2, otherwise as on the disclaimer slide</p>

<!--
Same laptop, same benchmark, a week later. Scoring got a lot faster: Uniform went from three times faster than SciPy to twelve.

[click] And Pietro measured again. In his words: round three goes to Francesco, no contest. That's at a million rows, on the cheapest formulas; at ten million rows it's still split. And the ball is in Nim's court, bouncing rather fast. What I can tell you for sure: a friend's blog post made polars-stats faster.
-->

---
eyebrow: 07 · Cost
---

# What polars-stats costs, and what comes next

<div class="cols">
<ul class="points">
<li v-click="1"><b>A much shorter catalogue</b>than SciPy's hundred-plus; no multivariate distributions, no statistical tests</li>
<li v-click="2"><b>No <code>fit</code>, no <code>loc</code> / <code>scale</code> shims</b>write the estimator as a Polars expression</li>
<li v-click="3"><b>One trip into Rust per method</b>even a closed-form mean, so validation stays uniform</li>
<li v-click="4"><b>A compiled dependency</b>that follows Polars' plugin interface across releases</li>
</ul>
<div class="cards">
<div v-click="5" class="card"><b>Where you need any of that</b>SciPy is the right tool, and the two live happily in one project</div>
<div v-click="6" class="card move"><b>Next: big plans</b>
<ul class="next">
<li v-click="7">every univariate distribution statrs supports</li>
<li v-click="8">multivariate distributions</li>
<li v-click="9">hypothesis testing</li>
</ul>
</div>
</div>
</div>

<p class="footnote">docs › Explanation › Why polars-stats › What it costs</p>

<!--
What it costs, straight from the docs. [click] A much shorter catalogue than SciPy's hundred-plus: no multivariate distributions, no statistical tests. [click] No fit, and no loc and scale shims. [click] One round trip into Rust per method call, even for a mean, so validation stays uniform. [click] And a compiled dependency that has to follow Polars' plugin interface across releases.

[click] Where you need any of that, SciPy is the right tool, and the two live happily in one project.

[click] So what's next? Big plans.

[click] First, every univariate distribution statrs supports. The thin layer is in place and tested on both engines, so each new one is mostly a file of short Rust functions and one row in a test registry.

[click] Then multivariate distributions.

[click] And hypothesis testing.
-->

---

# Every contract lives in the thin layer between the engine and the maths

<Stack :pins="4" summary />

<!--
Back to the picture. I wrote none of Polars' engine and none of statrs' maths, and SciPy gave me the vocabulary and the ruler. The strip in the middle carries four pins: reproducibility, broadcast, parameter validation, accuracy. Every bug that mattered lived there, in a promise nobody had made me.
-->

---
class: statement
---

<h1 class="lines">
<span>Borrow the engine.</span>
<span v-click="1">Borrow the maths.</span>
<span v-click="2">Enforce what neither promises you,</span>
<span v-click="2">inside the plugin, before it becomes a bug.</span>
</h1>

<!--
So if you take one thing home: borrow the engine. [click] Borrow the maths. [click] And enforce what neither of them promises you, inside your own code, before it becomes a bug. And when the bug is in theirs, send a reproducer upstream.
-->

---

# Thank you, SciPy, Polars and statrs

<div class="links">
<div>
<p class="install"><code>pip install polars-stats</code></p>
<p>Docs: <a href="https://fbruzzesi.github.io/polars-stats/">fbruzzesi.github.io/polars-stats</a>, start with <b>Explanation</b></p>
<p>Plugins: <a href="https://docs.pola.rs/user-guide/plugins/">docs.pola.rs/user-guide/plugins</a></p>
<p>statrs: <a href="https://docs.rs/statrs">docs.rs/statrs</a></p>
</div>
<figure class="qr">
<img src="/qr-repo.svg" alt="QR code for github.com/FBruzzesi/polars-stats">
<figcaption>polars-stats on GitHub</figcaption>
</figure>
<figure class="qr">
<img src="/qr-linkedin.svg" alt="QR code for linkedin.com/in/francesco-bruzzesi">
<figcaption>Connect on LinkedIn</figcaption>
</figure>
</div>

<!--
Thank you to the three giants: SciPy, Polars for the engine and the door into it, statrs for the maths. Thank you to the statrs contributors who turned my reports into fixes, and to the people who contributed to polars-stats.

If you want the why behind everything I showed, the docs have an **Explanation** section; this talk is its long version. Thank you.
-->
