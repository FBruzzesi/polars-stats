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
routerMode: hash
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
<Frame merged :cols="['sensor', 'reading', 'mu', 'sigma', 'z']" :ghost="$clicks >= 1 ? [] : ['z']" :flag="$clicks >= 1 ? 'reading' : undefined" fresh="z" />
<img v-click="1" src="/bells.svg" alt="One bell curve per sensor, the upper tail beyond a healthy reading shaded, each fault far off the chart">
<p v-click="1" class="caption">z: how far · the tail: how surprising</p>
</div>

<div v-click="1" class="caption">

$z = \dfrac{\texttt{reading} - \mu}{\sigma}$: how many standard deviations a reading sits from its own sensor's mean

</div>

---
class: section
eyebrow: 01 / Two libraries
---

# Polars has the data.<br>SciPy has the maths.<br>Why not use both?

<p class="subtitle">What each one gives you, and the glue most of us already write</p>

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
<li v-click="2"><b>Lazy</b>a plan first; the optimizer rewrites it before a row is read</li>
<li v-click="3"><b>Two engines</b><code>"in-memory"</code>, or <code>"streaming"</code> in small batches called morsels</li>
</ul>
</div>

<p v-click="4" class="callout">No probability distributions: a query engine, not a statistics package</p>

<p class="footnote">pola.rs · a dataframe engine written in Rust</p>

---
eyebrow: 01 · SciPy
---

# `scipy.stats`: a hundred distributions, one vocabulary

<table class="plain">
<thead><tr><th>pdf</th><th>cdf</th><th>sf</th><th>ppf · isf</th><th>logpdf · logsf</th><th>rvs</th></tr></thead>
<tbody><tr><td>density</td><td>P(below)</td><td>P(above)</td><td>back to a value</td><td>in logs</td><td>samples</td></tr></tbody>
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

<p v-click="2" class="callout">Broadcasting: a single value stretches to the arrays' length</p>

<p class="footnote">scipy.org · Python's scientific library</p>

---

# The round trip works, and the lazy query ends at `collect`

<div class="cols wide-left small-code">
<div class="window" data-file="round_trip.py" data-lang="Python">

```python {all|4|5-6,9-12|all}{at:1}
import polars as pl
from scipy import stats

df = pl.scan_parquet("readings.parquet").collect()
cols = df.select("mu", "sigma", "reading")
mu, sigma, x = cols.to_numpy().T
baseline = stats.norm(loc=mu, scale=sigma)

df = df.with_columns(
    upper_tail=pl.Series(baseline.sf(x)),
    simulated=pl.Series(baseline.rvs(random_state=42)),
)
```

</div>
<div class="cards">
<div v-click="1" class="card"><b>The query ends</b>the optimizer stops at <code>collect</code></div>
<div v-click="2" class="card"><b>Alignment is yours</b>bare arrays out, bare arrays in</div>
<div v-click="3" class="card bad"><b>NumPy has no null</b>missing and invalid both become <code>NaN</code></div>
</div>
</div>

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

---
class: section
eyebrow: 02 / The plugin
---

# Can a distribution be a Polars expression instead?

<p class="subtitle">Keep the query whole, and keep every answer on its own row</p>

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
<div v-click="1" class="card bad"><b>Polars' own <code>erfc</code></b>one tail of one distribution</div>
<div v-click="2" class="card bad"><b>Back to Python: <code>map_elements</code></b>a Python call per row, on one thread</div>
<div v-click="3" class="card move"><b>An expression plugin</b>compiled Rust, called like a native expression</div>
</div>

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

<p class="footnote">docs.rs/statrs · a Rust crate</p>

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

---
eyebrow: 02 · The plugin
---

# Python plans, Rust walks the rows, statrs computes

<div class="cols anatomy small-code">
<div class="window" data-file="polars_stats/distributions/_base.py, simplified" data-lang="Python">

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
<div class="window" data-file="src/distributions/normal.rs, simplified" data-lang="Rust">

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

<p v-click="4" class="callout"><code>is_elementwise=True</code>: output row i depends only on input row i</p>

---
eyebrow: 02 · The plugin
---

# One flag lets the optimizer move the filter

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

---

# Polars runs it, statrs computes it. What is left for me to write?

<Stack :pins="0" />

---
class: section
eyebrow: 03 / Reproducibility
---

# Same seed, same column?

<p class="subtitle">The second task: simulated readings, identical on every run</p>

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

---
eyebrow: 03 · Reproducibility
---

# A seeded column repeats across runs and engines

<div class="cols small-code">
<div class="window" data-file="simulate.py" data-lang="Python">

```python
import polars as pl
import polars_stats as ps

baseline = ps.Normal(mu="mu", sigma="sigma")
draws = baseline.sample(seed=42)
sim = readings.with_columns(simulated=draws)
a = sim.collect(engine="in-memory")
b = sim.collect(engine="streaming")
a.equals(b)  # True
```

</div>
<div v-click="1"><Frame :cols="['sensor', 'mu', 'sigma', 'simulated']" fresh="simulated" /></div>
</div>

<Stack v-click="2" mini :pins="1" />

---
class: section
eyebrow: 04 / Broadcast
---

# Does every input arrive at full length?

<p class="subtitle">NumPy stretches a single value. What reaches a plugin?</p>

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

---
class: section
eyebrow: 05 / Parameter validation
---

# Does every row reach the parameter check?

<p class="subtitle">SciPy answers an impossible parameter with NaN. We wanted an error</p>

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
<tbody>
<tr><td>a null value or a null parameter</td><td><code>null</code></td></tr>
<tr><td>a <code>NaN</code> value</td><td><code>NaN</code>, as in SciPy</td></tr>
<tr><td>an invalid parameter, on any row</td><td><code>ComputeError</code>: the whole query fails</td></tr>
</tbody>
</table>

<p v-click="3" class="callout">An expression cannot raise per row: the check is a plugin that sees every row</p>

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
    for i in 0..x.len() {  // illustrative, not the source
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

---
class: section
eyebrow: 06 / Accuracy
---

# Is the number on each row right?

<p class="subtitle">Every method agrees with SciPy. Is that enough?</p>

---
eyebrow: 06 · Accuracy
---

# In the far tail, agreeing with SciPy proves nothing

<ul class="points">
<li><b>Parity</b>every method, against SciPy, on the same inputs</li>
<li><b>Property tests</b>a cdf stays in [0, 1] and never goes down</li>
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

---
clicks: 1
eyebrow: 06 · Accuracy
---

# `log_sf` ranks the faults that `sf` rounds to zero

<div class="cols frame-left middle small-code">
<Frame :cols="['sensor', 'reading', 'upper_tail', 'log_tail']" :ghost="$clicks >= 1 ? [] : ['log_tail']" wrong="upper_tail" fresh="log_tail" />
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

---
clicks: 3
eyebrow: 06 · Accuracy
---

# The answer fits in a float64. The algorithm lost it

<div class="cols">
<div>
<p class="side-label">polars-stats: a coin with <code>p = 1e-300</code></p>
<div class="window" data-file="bernoulli.py" data-lang="Python">

```python
# P(value > 0.5), should be 1e-300
ps.Bernoulli(p=1e-300).sf(0.5)
```

</div>
<div v-click="1" class="derive compact">
<div><span>returned</span><code class="alarm">0.0</code></div>
<div><span>computed as</span><code>1 - (1 - p)</code></div>
<div><span>in float64</span><code>1 - 1e-300 == 1.0</code></div>
</div>
</div>
<div>
<p class="side-label">statrs: <code>0.5</code> by symmetry</p>
<div class="window" data-file="main.rs, statrs 0.19.1" data-lang="Rust">

```rust
Beta::new(s, s)?.cdf(0.5)
```

</div>
<table v-click="2" class="plain">
<thead><tr><th>s</th><th>statrs</th></tr></thead>
<tbody>
<tr><td>1e6</td><td class="warn">0.4912</td></tr>
<tr><td>1e7</td><td class="warn">0.2129</td></tr>
<tr><td>1e8</td><td class="alarm">−1.147</td></tr>
</tbody>
</table>
<p v-click="3" class="caption">reported upstream as statrs #434, fixed by another contributor in #456</p>
</div>
</div>

---
eyebrow: 06 · Accuracy
---

# A defect has two acceptable outcomes, and no third

<blockquote class="quote">The algorithm gets fixed, or the caveat gets documented with a regime and a magnitude. A runtime warning is never the fix, because it cannot fire per row from inside the engine.</blockquote>

<Stack v-click="1" mini :pins="4" />

---
class: section
eyebrow: 07 / Cost
---

# Was it worth it, and what does it cost?

<p class="subtitle">Before any speed number, everything wrong with the measurement</p>

---
eyebrow: 07 · Cost
---

# Every number that follows comes from one laptop

<div class="stamp">Disclaimer</div>

<dl class="conditions">
<dt>Machine</dt><dd>Apple M5, 10 cores (4 performance, 6 efficiency), 24 GB, macOS 26.7.2, on AC power, other apps closed</dd>
<dt>Software</dt><dd>Python 3.14.10 · Polars 2.1.0 · SciPy 1.18.1 · NumPy 2.5.4 · polars-stats 0.2.0, release build</dd>
<dt>polars-stats</dt><dd>streaming engine, default thread pool (10 threads); the timed call includes building and dispatching the query</dd>
<dt>SciPy</dt><dd>the classic frozen API, <code>norm(loc=…, scale=…)</code> then the method; samplers get a PCG64 <code>Generator</code>; one thread</dd>
<dt>Regime</dt><dd>one parameter value per row, at 10k, 1M and 10M rows; plain numbers and one-value expressions are other code paths on both sides, not measured here</dd>
<dt>Not measured</dt><dd>SciPy's newer API (measured earlier on the Normal, column parameters: 1.3 to 1.9× faster than the frozen one); peak memory</dd>
<dt>samples</dt><dd>in the column regime SciPy draws <code>(draws, rows)</code> and transposes a view: optimistic for SciPy by about one pass over the output</dd>
<dt>Timing</dt><dd>median of up to 50 iterations within 10 s per cell (at least 3), after one warmup · ratio = SciPy / polars-stats, above 1 means polars-stats is faster</dd>
</dl>

---
clicks: 4
eyebrow: 07 · Cost
---

# One value per row: large frames favour polars-stats

<div class="chart">
<img src="/after/bench-column.svg" alt="Dot plots of the SciPy to polars-stats time ratio per distribution, column regime, at 10k, 1M and 10M rows, for sample, samples and sf">
<img v-click="4" src="/after/bench-column-10k.svg" alt="Overlay: the 10k-row cells where SciPy is ahead">
<span class="dim sample" :class="{ on: $clicks === 2 || $clicks === 3 }"></span>
<span class="dim samples" :class="{ on: $clicks === 1 || $clicks === 3 }"></span>
<span class="dim sf" :class="{ on: $clicks === 1 || $clicks === 2 }"></span>
</div>

<p v-click="4" class="caption center">10k rows: SciPy ahead on <code>sample</code> for all but DiscreteUniform, up to 6×; polars-stats pays about 0.3 ms a call</p>

<p class="footnote">*SciPy slow paths: <code>randint</code> with per-row bounds runs <code>np.vectorize</code>; Bernoulli is <code>binomial(1, p)</code></p>

---
eyebrow: 07 · Cost
---

# What polars-stats costs, and what comes next

<div class="cols">
<ul class="points">
<li v-click="1"><b>A much shorter catalogue</b>no multivariate, no statistical tests</li>
<li v-click="2"><b>No <code>fit</code>, no <code>loc</code> / <code>scale</code> shims</b>write the estimator as a Polars expression</li>
<li v-click="3"><b>One trip into Rust per method</b>even for a closed-form mean</li>
<li v-click="4"><b>A compiled dependency</b>tracks Polars' plugin interface</li>
</ul>
<div class="cards">
<div v-click="5" class="card"><b>Where you need any of that</b>SciPy is the right tool</div>
<div v-click="6" class="card move"><b>Next: big plans</b>
<ul class="next">
<li v-click="7">every univariate distribution statrs supports</li>
<li v-click="8">multivariate distributions</li>
<li v-click="9">hypothesis testing</li>
</ul>
</div>
</div>
</div>

---

# Every contract lives in the thin layer between the engine and the maths

<Stack :pins="4" summary />

---
class: statement
---

<h1 class="lines">
<span>Borrow the engine.</span>
<span v-click="1">Borrow the maths.</span>
<span v-click="2">Enforce what neither promises you,</span>
<span v-click="2">inside the plugin, before it becomes a bug.</span>
</h1>

---

# Thank you, SciPy, Polars and statrs

<div class="links">
<div>
<p class="install"><code>pip install polars-stats</code></p>
<p>Docs: <a href="https://fbruzzesi.github.io/polars-stats/">fbruzzesi.github.io/polars-stats</a></p>
<p>Polars plugins: <a href="https://docs.pola.rs/user-guide/plugins/">docs.pola.rs/user-guide/plugins</a></p>
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

---
eyebrow: Appendix · Cost
---

# Two days later, a friend rewrote polars-stats in Nim

<div class="cols wide-left middle">
<div class="cards">
<div v-click="1" class="card"><b>"Polars plugins, without the Rust"</b>14 distributions in about 780 lines of Nim: on par</div>
<div v-click="2" class="card move"><b>So I went back to work</b>a vectorised parameter check, one pass per answer</div>
</div>
<figure class="qr">
<img src="/qr-nanuq.svg" alt="QR code for pietroppeter.github.io/nanuq-more">
<figcaption>pietroppeter.github.io/nanuq-more</figcaption>
</figure>
</div>

<p class="footnote">line counts and verdict from the post · both changes ship in polars-stats 0.2.0</p>

---
eyebrow: Appendix · Cost
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

<p class="footnote">the same run as the benchmark slide</p>
