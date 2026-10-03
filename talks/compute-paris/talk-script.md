# Vectorized statistical distributions, built on the Polars engine

**Destination (one sentence):** Two giants hand you the engine and the maths, and every bug that mattered lived in
the thin layer between them, in the promises neither giant ever made: the order rows arrive in, how long each input
is, which rows actually run, and how many digits survive.

**The lesson, as one instruction:** borrow the engine, borrow the maths, and enforce what neither promises you inside
the plugin, before it becomes a bug.

**Smallest concrete case:** two temperature sensors from the docs tutorial. `temp-a` sits around 10 degrees
(`mu = 10`, `sigma = 0.5`), `temp-b` around 100 (`mu = 100`, `sigma = 2`). Six readings, two of them faults (`40.0` on
`temp-a`, `250.0` on `temp-b`). Two jobs, which were my own first use case: **simulate** readings per sensor
(`sample`, `samples`) and **score** real ones against their own sensor's baseline (z-score, then tail probability,
`sf`). Where it stops transferring: real telemetry has drift and non-normal noise; the mechanics do not care.

**Visual spine: the stack.** Polars on top (the engine), statrs at the bottom (the maths), and between them a thin
strip labelled "polars-stats". scipy stands beside the stack as the ruler. Each beat pins one promise onto the thin strip:
*order*, *length*, *every row*, *digits*. On the right, the six-row sensor frame is the running example and gains one
column per beat (`z`, `upper_tail`, `simulated`, `log_tail`). The closing slide is the opening stack, now carrying all
four pins.

**Audience / assumed knowledge / slot:** compute! Paris 2026, people who work with open-source computation and data,
across languages. Assume they know what a table, a function and a probability are. Assume nothing about Python, scipy,
Polars, Rust or statrs: each is introduced at the moment the story needs it, and each gets its credit there.
25 minutes plus 5 of Q&A. No live demo: every output on a slide is a recorded, re-run result (see Sources).

**Promise (said in the first minute):** after this talk you can build on someone else's engine and someone else's
maths without being an expert in either, and you will know where your own bugs are going to live.

**House rules for this script:** never a count of the distributions shipped (it will change before November); sensor
telemetry only; neutral tone towards scipy (state what each returns and why, never "more correct"); benchmarks only
after their disclaimers.

## Question chain

1. Each sensor has its own baseline. How do I simulate readings and score real ones, row by row, on a dataframe?
2. Can the distribution live inside the Polars query instead?
3. If the engine and the maths are borrowed, what is left for me to write?
4. The plugin sees pieces. Do the inputs arrive at full length, and does every row run?
5. Every row runs now. Is the number on each row right?
6. Was it worth it, and what does it cost?

Retelling test (what an attendee says at lunch): "The speaker built a stats library on Polars and a Rust maths crate
without really knowing Rust. The engine and the maths came for free; every bug was in the little bit of glue in between,
in things nobody had promised."

## Timing

| Beat | Question (short) | Minutes | Ends at |
|---|---|---|---|
| 1 | How do I simulate and score per row? | 4.0 | 4:00 |
| 2 | Can it live inside the query? | 4.5 | 8:30 |
| 3 | What is left for me to write? | 4.25 | 12:45 |
| 4 | Full length, every row? | 4.25 | 17:00 |
| 5 | Is the number right? | 4.5 | 21:30 |
| 6 | Worth it, and the cost? | 4.5 | 26:00 |

The full script is about 3,400 spoken words: 26 minutes at 130 words per minute, 25 at 136. Cut list items 1 to 3
take off about a minute more. Rehearse with a timer: if beat 2 ends past 9:00, take them.

## Beats

### Beat 1: Each sensor has its own baseline. How do I simulate readings and score real ones, row by row, on a dataframe?

**Question:** the one the room already has: how do I do statistics on data that lives in a dataframe?

**Natural attempt:** leave the dataframe. Take the columns to NumPy, call `scipy.stats`, put the results back. And it
is a good attempt: scipy broadcasts parameter arrays, so a different distribution per row is already vectorised.

**Wall:** the cost is where the result lands. The lazy query has to end before you can leave it, keeping the result
attached to the right rows is now your job, and an invalid parameter comes back as `nan` with no warning.

**Move:** want the result to stay an expression inside the query. (The how is beat 2.)

**Picture:**

1. The six-row frame, `sensor` and `reading` only. Question on the slide: "Two of these are faults. Which?"
2. Click: the baselines appear as `mu` and `sigma` columns, and a `z` column fills in (`z = (reading - mu) / sigma`).
3. Click: a bell curve per sensor, with the tail beyond each reading shaded. Label: "how far" (z) vs "how surprising"
   (tail probability).
4. The scipy round trip, as code:

    ```python
    df = lf.collect()                          # the lazy query ends here
    mu, sigma, x = (df[c].to_numpy() for c in ("mu", "sigma", "reading"))
    rng = np.random.default_rng(42)
    df = df.with_columns(
        simulated=pl.Series(stats.norm(loc=mu, scale=sigma).rvs(random_state=rng)),
        upper_tail=pl.Series(stats.norm(loc=mu, scale=sigma).sf(x)),
    )
    ```

5. Click: three labels appear next to the code: "the query ends", "alignment is yours", "?".
6. Quiz slide: `stats.norm(loc=0, scale=-1).sf(1.0)` returns what? Click: `nan`. Footnote: no warning emitted.
7. The stack appears for the first time, nearly empty: only scipy, standing at the side, labelled "the vocabulary,
   and the ruler".

**Terms introduced here:** z-score, tail probability, Polars (one sentence), lazy query, scipy, `scipy.stats`, the
method vocabulary (`pdf`, `cdf`, `sf`, `ppf`, `isf`, `rvs`).

**Say:**

Here are six temperature readings from two sensors. Sensor A usually sits around 10 degrees, give or take half a
degree. Sensor B sits around 100, give or take 2. Two of these readings are faults. Which ones? [pause]

Forty on sensor A, two hundred and fifty on sensor B. You just did that in your head with a z-score: how many standard
deviations a reading sits from its sensor's mean. Forty is sixty standard deviations above ten.

[click] That tells you how far. What I usually need is how surprising: if the sensor is healthy, how likely is a
reading at least this high? That's the tail of the distribution, and it lets you set an alarm at "one in a thousand"
instead of at a magic number of degrees, different for every sensor.

And before I score real readings, I want fake ones: simulated telemetry for each sensor, from its own baseline, one
draw per row or a thousand per row. So, two jobs: sample and score. And every row has its own distribution.

I'm Francesco. One thing that matters for the next 25 minutes: I'm not a Rust expert, and a good part of the Rust in
polars-stats, the library this talk is about, was written with AI assistance. This talk is about what you can build
anyway, standing on the right shoulders, and by the end you'll know where your own bugs are going to live.

My data lives in Polars, a dataframe library written in Rust, with a Python API. Its trick is laziness: you describe
the whole query first, and Polars plans it before running any of it.

The first pair of shoulders is scipy, Python's scientific library, around since 2001. Its statistics module,
`scipy.stats`, has more than a hundred distributions, and they all speak one vocabulary: `pdf` for the density, `cdf`
for the probability below a value, `sf`, the survival function, for the probability above it, `ppf` and `isf` to go
from a probability back to a value, and `rvs` to draw samples. polars-stats copies that vocabulary, word for word.

And scipy already solves my problem. Give it an array of means and an array of standard deviations, and it scores every
element against its own distribution, in compiled code, with no Python loop. So the honest first attempt is this one.
[click] Take the columns out, give them to scipy, put the results back. It works. I did it for years.

What it costs is where the result lands. [click] First, the query: to leave a lazy query you have to run it, so
everything after this line is a second query, and the optimiser never sees the two together. Second, alignment: the
result is a bare array, and keeping it on the right rows through the next join is your job. Third, a quiz. [click] A
normal distribution with a standard deviation of minus one. That's not a distribution. What does scipy give you for
its tail? [pause]

[click] `nan`. No warning. A modelling error, travelling down your pipeline dressed up as missing data.

None of this is scipy's fault: an array has no query plan, no row identity and no null. [click] But I wanted the
answer to stay inside the query.

**Forces next:** Can the distribution live inside the Polars query instead?

---

### Beat 2: Can the distribution live inside the Polars query instead?

**Question:** Can the distribution live inside the Polars query instead?

**Natural attempt:** Polars computes the z-score natively, so write the rest in Polars too. When that runs out, call
Python per row with `map_elements`, or type in a textbook approximation of the error function.

**Wall:** the tail probability needs a special function, the error function `erf`, and Polars has none (checked on
Polars 1.44.2: no `erf` on `pl.Expr`). `map_elements` runs one row at a time through the interpreter on one thread. A
hand-typed approximation is fine in the middle and falls apart in the tails, which is exactly where anomaly detection
lives.

**Move:** an expression plugin (a compiled Rust function that Polars treats as one of its own expressions) for the
engine side, and statrs for the maths. One flag, `is_elementwise=True`, and Polars hands you the rest.

**Picture:**

1. The frame with `z` computed in plain Polars: `(pl.col("reading") - pl.col("mu")) / pl.col("sigma")`. Next to it,
   the formula for the tail, with `erf` in red: "Polars: no `erf`".
2. Click: the three attempts listed, each with its one-line cost.
3. Click: the stack fills in. Polars box on top ("the engine"), statrs box at the bottom ("the maths"), and between
   them a thin, empty strip labelled "polars-stats".
4. The anatomy, three layers top to bottom, one line each:

    ```python
    baseline = ps.Normal(mu="mu", sigma="sigma")      # Python: returns pl.Expr, computes nothing
    lf.with_columns(upper_tail=baseline.sf("reading"))
    ```

    ```rust
    #[polars_expr(output_type=Float64)]
    fn normal_sf(inputs: &[Series]) -> PolarsResult<Series> {
        value_keyed(inputs, sf_value)                 // Rust: columns in, column out
    }

    fn sf_value(dist: &Normal, v: f64) -> Option<f64> {
        Some(dist.sf(v))                              // statrs: the maths
    }
    ```

5. Click: one line highlighted in the Python registration, `is_elementwise=True`, with the caption "a promise: row i
   of the output depends only on row i of the inputs".
6. The pipeline, then its real `explain()` output (Polars 1.44.2, a 1,000,000-row Parquet file):

    ```python
    scored = (
        pl.scan_parquet("readings.parquet")
        .with_columns(upper_tail=ps.Normal(mu="mu", sigma="sigma").sf("reading"))
        .filter(pl.col("sensor") == "temp-b")
        .filter(pl.col("upper_tail") < 1e-3)
    )
    print(scored.explain())
    ```

    ```text
    FILTER col("upper_tail") < 0.001
    FROM
       WITH_COLUMNS:
       [col("reading").…/_internal.abi3.so:normal_sf([col("mu"), col("sigma")]).alias("upper_tail")]
        Parquet SCAN [readings.parquet]
        PROJECT */4 COLUMNS
        SELECTION: col("sensor") == "temp-b"
    ```

    Highlight `SELECTION: col("sensor") == "temp-b"` inside the scan, below the plugin node.
7. Click: `collect(engine="in-memory")` and `collect(engine="streaming")`: 482 rows each, `equals` is `True`. Then
   one line each for `sink_parquet`, `over` / `group_by`, and "all 10 cores".
8. Back to the stack: Polars and statrs labelled with what each gave; the strip between them still empty.

**Terms introduced here:** `erf`, `map_elements`, the GIL (one sentence), expression plugin, Rust crate, statrs,
`is_elementwise`, predicate pushdown (shown, then named), streaming engine, morsel.

**Say:**

Polars can do the first half on its own. [click] Reading minus mu, divided by sigma: plain arithmetic, on every core.
The next step is the problem. Turning a z-score into a tail probability needs a special function, the error function,
`erf`, and Polars doesn't have one. Fair enough: it's a dataframe engine, not a statistics package.

So what would you try? [pause] [click] Call a Python function once per row with `map_elements`: it works, one row at a
time, through the interpreter, on one thread. Type a textbook approximation of `erf` as Polars arithmetic: fine in the
middle, wrong in the tails, which is exactly where anomaly detection lives.

The third option is the one Polars offers itself: an expression plugin. You write a function in Rust, compile it, and
Polars calls it as if it were one of its own expressions. The function receives columns and returns a column, and that
is the whole contract.

That leaves the maths, and I was not going to write an error function. [click] Meet the second pair of shoulders:
statrs. It's a Rust crate, which is Rust's word for a library, and it's roughly `scipy.stats` for Rust. Build
`Normal::new(mu, sigma)` and you get the density, the cdf, the survival function, the inverse cdf, the log density and
sampling, with `erf`, gamma and beta underneath. It's maintained by volunteers, and it's the reason polars-stats
exists.

[click] So here's the whole anatomy. On top, a Python class whose parameters can be column names. Its methods compute
nothing: each one returns a Polars expression that says "call this Rust function on these columns". In the middle, the
Rust function: it gets the columns, walks the rows, builds a distribution per row. At the bottom, one line:
`dist.sf(v)`. That line is statrs.

It wasn't always this plain: the first versions generated these functions with Rust macros, code that writes code, and
I couldn't review what they produced. When you're not an expert in a language, code you can read beats code that saves
typing.

Now the part I enjoy the most. [click] When I register the function with Polars, I pass one flag:
`is_elementwise=True`. It's a promise: row i of the output depends only on row i of the inputs. In exchange for that
one promise, Polars gives me everything else. Watch.

[click] Here's a pipeline. Scan a Parquet file of a million readings, add the tail probability, keep sensor B, keep the
readings rarer than one in a thousand. And here's what Polars plans to do, read from the bottom up. [pause] The filter
on sensor B, which I wrote after the scoring, has moved down into the file reader. Half the rows are never decoded, so
they never reach my function. The filter on the score stays on top, because it needs the score. That's called
predicate pushdown, and I wrote none of it.

[click] Run it on the in-memory engine, and on the streaming engine, which reads the file in small batches called
morsels, so the data never has to fit in memory. Same 482 rows, identical. Swap `collect` for `sink_parquet` and it
streams from file to file. Put it inside a `group_by` or an `over` and it works per group. All of it on Polars' thread
pool, across every core.

I wrote a loop over rows. Polars made it lazy, optimised, parallel, streaming and groupable. statrs made the numbers
right. scipy told me what right looks like.

[click] So the picture is this: Polars on top, statrs at the bottom, scipy beside them holding the ruler. And between
Polars and statrs, a thin strip of code that I actually wrote. That strip is polars-stats.

**Forces next:** If the engine and the maths are borrowed, what is left for me to write?

---

### Beat 3: If the engine and the maths are borrowed, what is left for me to write?

**Question:** If the engine and the maths are borrowed, what is left for me to write?

**Natural attempt:** the first commit. One distribution (`Bernoulli`, a biased coin), one method (`sample`): seed one
random number generator, walk the rows in order, draw one value per row with statrs. Its docstring is honest about the
limit: chunked and streaming inputs are "not supported".

**Wall:** "not supported" is not a mode the caller can switch off. Polars hands a plugin pieces of a column (chunks,
thread splits, streaming morsels, groups), each call restarts the generator from the same seed, and the result depends
on where the pieces were cut.

**Move:** key every draw on `(seed, row index)`, pass the row index in as an ordinary column so Polars carries it, and
pay for it with a generator that is cheap to build (`Pcg64Mcg`). Where statrs samples a Binomial in O(n), borrow from
`rand_distr` instead. And notice the gap this fills: no Polars-native alternative gave a seeded column that survives
how Polars cuts the data.

**Picture:**

1. The first commit, dated 26 April 2026: the Rust loop condensed to five lines (seed one `ChaCha20Rng`, `for i in
   0..n`, draw), and the Python docstring sentence verbatim: "Reproducibility under ``seed`` assumes a single-chunk
   input series; chunked / streaming inputs are not supported."
2. A column drawn as one long bar, then cut into pieces three ways: "chunks", "threads", "morsels (streaming)". Each
   piece gets its own arrow labelled "generator restarts from seed 42".
3. Prediction slide: "Same frame, same seed, in-memory vs streaming. Same numbers?" Click: "No: the seed reproduced
   the chunk layout, not the query."
4. The move as a picture: every row gets its own small die, labelled `(42, 0)`, `(42, 1)`, `(42, 2)` … The pieces
   can be cut anywhere and each die stays the same.
5. The bill: "one `ChaCha20` per row: 10 to 20 times slower" (crossed out) then "`Pcg64Mcg`: about 5.9 s to 0.29 s
   on 1M rows × 100 draws".
6. The Binomial aside: "statrs: n coin flips per draw, O(n)" vs "`rand_distr`: cost flat in n".
7. Today, one line: a seeded column is identical across runs, chunk layouts, thread counts and both engines; CI runs
   every test once per engine.
8. The alternatives, as a small table, credited for what they do well (sampling-first APIs; polars_rng's wider
   sampling catalogue):

    | `seed=42`, 1M rows, Polars 1.44.2 | re-run | in-memory vs streaming | 1 chunk vs 4 chunks |
    |---|---|---|---|
    | polars_rng | no `seed` argument | | |
    | polars-random 0.5.0 | same | different | different (column parameters) |
    | polars-stats 0.1.0 | same | same | same |

    Footnote: "measured 2026-10-03". Then the stack's thin strip gets its first pin: **order**. The frame gains
    `simulated`.

**Terms introduced here:** seed, random number generator, chunk, thread pool, morsel (recalled), `ChaCha20`,
`Pcg64Mcg`, `rand_distr`, `samples`.

**Say:**

The glue started very small.

[click] 26 April 2026, the first commit of polars-stats. One distribution, Bernoulli: a biased coin that comes up 1
with probability p. One method, `sample`. Under seventy lines of Rust. The idea is what anyone would write: take the
seed, start one random number generator, walk the rows in order, and draw one value per row with statrs.

And the docstring was honest. [click] "Reproducibility under seed assumes a single-chunk input series; chunked or
streaming inputs are not supported."

Not supported. As if that were a mode the caller could switch off.

[click] Polars doesn't hand a plugin "the column". It hands it pieces. A column can
live in several chunks, the thread pool splits work across cores, the streaming engine sends morsels, a group by sends
one group at a time. My function runs once per piece, and every call restarts the generator from the same seed.

So, a prediction. [pause] Same frame, same seed, in-memory engine against streaming engine. Same numbers? [pause]
[click] No. The first rows of every piece repeat each other, and where the pieces start depends on the engine, not on
your data. The seed was reproducing the chunk layout, not the query.

That one never reached a release, but it changed how I think about the plugin. It sees whatever pieces Polars
decides to give it.

[click] So the move: stop thinking of one stream walking down the rows. Give every row its own generator, seeded from
two numbers, the seed and the row's index. Row 7 draws from the stream for 42 and 7, whichever piece, thread or engine
it lands in. And the row index is an ordinary input column, so Polars carries it through chunks, morsels and groups
for me.

That fix came with a bill. [click] My generator, ChaCha20, is a cryptographic one, and building one per row made
sampling ten to twenty times slower. The fix for the fix was `Pcg64Mcg`, which costs a handful of integer operations to
build. On that day's benchmark, a million rows times a hundred draws went from about 5.9 seconds to 0.29. And `samples`,
a thousand draws per row, is the first thousand values of that row's own stream.

[click] One place where a giant said no: statrs draws a Binomial by flipping n coins, so n random numbers per draw.
Binomial sampling borrows from another crate instead, `rand_distr`, whose cost doesn't grow with n. Everything else
about the Binomial still comes from statrs.

[click] Today a seeded column repeats across runs, chunk layouts, thread counts and both engines, and CI runs every
test once per engine.

[click] And this turned out to be the one thing I couldn't find anywhere else. There are two good Polars plugins for
sampling, polars-random and polars_rng, and for pure simulation polars_rng's catalogue is wider than polars-stats'. But
polars_rng takes no seed at all. polars-random takes one, and on its latest release the same seed gives one column on
the in-memory engine and a different one on streaming. For simulation, that's a reasonable trade. For testing a
pipeline, I needed the seeded column to belong to the data, not to how Polars happened to cut it.

First pin on the strip: order.

**Forces next:** The plugin sees pieces. Do the inputs arrive at full length, and does every row run?

---

### Beat 4: The plugin sees pieces. Do the inputs arrive at full length, and does every row run?

**Question:** The plugin sees pieces. Do the inputs arrive at full length, and does every row run?

**Natural attempt (length):** the first commit already knew Polars does not stretch a one-value input before the plugin
runs, so the Python side padded every plain number to full length. **Natural attempt (rows):** since a Polars
expression cannot raise an error per row, route the parameters through a tiny validating plugin, placed inside a
`when / then / otherwise` that handles nulls.

**Wall (length):** `pl.lit(0.5)`, a column's mean or its first value are one row long only at run time, Python cannot
see that, and the Rust helper that walks several columns stops at the shortest. PyPI 0.0.1, one million rows,
`sigma=pl.lit(0.5)`: one distinct answer on the in-memory engine, ten on streaming, right height, no error.
**Wall (rows):** Polars 1.44 evaluates each branch only on the rows that select it. The validator stopped seeing the
rows it existed to reject: PyPI 0.0.1 returns `0.0` for a Uniform whose maximum is below its minimum on Polars 1.44,
and raises on 1.43.

**Move:** alignment by length, in Rust, as the first thing every plugin does (breaking release 0.0.2). Then: cap Polars
below 1.44, move every closed form that branches on the value into Rust, lift the cap, delete the Python null wrapper
that had the same leak. The plugin owns every row.

**Picture:**

1. The second comment from the first commit, quoted (shortened): "`is_elementwise=True` does NOT broadcast inputs
   before the plugin runs … a length-1 `pl.lit(p)` would make the plugin draw a single sample which polars then
   repeats across the whole frame."
2. Three ways to write "0.5": `0.5`, `pl.lit(0.5)`, `pl.col("reading").mean()`. The first is padded in Python (green);
   the other two are one row long only at run time (red, with "Python cannot see this").
3. Prediction slide: PyPI 0.0.1, `Normal(mu=0.0, sigma=pl.lit(0.5)).cdf("x")` on one million rows. "The right answer
   has 1,000,000 distinct values. How many do we get?" Click: in-memory **1**, streaming **10**. Height 1,000,000 on
   both. No error.
4. The move: `align_inputs`, five lines of pseudo-code: "if an input has length 1, stretch it to the others' length;
   if two lengths disagree, raise". Caption: "by length at run time, not by kind of expression". Footnote: breaking
   release 0.0.2.
5. Callback, the payoff of the fix: estimate each sensor's baseline from its own readings and score against it, in
   one expression:

    ```python
    r = pl.col("reading")
    lf.with_columns(upper_tail=ps.Normal(mu=r.mean(), sigma=r.std()).sf(r).over("sensor"))
    ```

    Caption: "`r.mean()` is one value per group: length 1, stretched by the plugin". Footnote: "0.0.1: `ShapeError` on
    the in-memory engine. Today: both engines."
6. The validation shape, as a diagram: `when(null).then(null).otherwise(validate(params) → formula)`.
7. Polars 1.44, one line: "each branch is evaluated only on the rows that select it". Then the same published wheel
   (0.0.1), the same frame (`min = 1`, `max = 0`, `x = -5`), two Polars versions: 1.43.2 raises `ComputeError`, 1.44.2
   returns `0.0`.
8. The ten-day fix as four steps on a line: cap `polars<1.44` · move the closed forms into Rust · lift the cap ·
   delete the Python null wrapper. The strip gets two pins: **length**, **every row**.

**Terms introduced here:** broadcasting, `pl.lit`, run time vs plan time, `when / then / otherwise`, `ComputeError`,
breaking release.

**Say:**

Here's another comment from that first commit. [click] Roughly: Polars doesn't broadcast inputs into a plugin, so a
one-value `pl.lit(p)` would draw a single sample that Polars then repeats across the frame. Broadcasting means
stretching one value to the length of the column. So Python stretched every plain number before calling Rust. Handled.

It handled the case I thought of. [click] But Polars also lets you write `pl.lit(0.5)`, or a column's mean, or its
first value. Each is one row long, and only the running engine knows it, not the Python that builds the query. And the
Rust helper that walks several columns together stops at the shortest one.

Prediction time. [click] Version 0.0.1, straight from PyPI. A million rows, a Normal with sigma equal to
`pl.lit(0.5)`, scored with `cdf`. The right answer has a million distinct values. How many do we get? [pause]

[click] In-memory: one. Row zero's answer, a million times. Streaming: ten, one per morsel. The right height, no
error, no warning. That one I published.

[click] The move: align in Rust, not in Python. The first thing every plugin does now is stretch any input of length one
to the others' length, and raise if two real lengths disagree, judged by actual length at run time, not by kind of
expression. It changed what some calls return, so it shipped as a breaking release, 0.0.2.

[click] That fix is what makes this line work. It estimates each sensor's baseline from its own readings and scores
every reading against it, in one expression, inside an `over`. The mean and the standard deviation are one
value per sensor: length one. The plugin stretches them, Polars does the grouping. On 0.0.1 this line raised a shape
error on the in-memory engine. Today it runs on both. Second pin: length.

Now, which rows run. Remember scipy's `nan` for a negative standard deviation? I wanted an error instead. But a Polars
expression can't raise on a bad row. So for the methods simple enough to write as Polars arithmetic, the parameters went
through a tiny Rust plugin whose only job was to check and raise. [click] And that check sat inside a
`when / then / otherwise`, Polars' if-else, next to the null handling.

[click] Then Polars 1.44 shipped a good optimisation: evaluate each branch only on the rows that select it. For Polars'
own expressions, that's pure win. For my validator, it meant it no longer saw the rows it existed to reject.

Same published wheel, same three-row frame: a Uniform whose maximum is below its minimum. On Polars 1.43: an error, as
designed. On Polars 1.44: [pause] zero. A quiet number where an error should be. And a plain `pip install` would give
you exactly that pair.

[click] The fix took about ten days. Cap Polars below 1.44, so nobody upgrades into it. Move every closed form that
branches on the value into Rust, so the check runs on every row by construction. Lift the cap. Then find the same leak
one level up, in a Python null wrapper, and delete it. I opened an issue upstream too; it's still open, and that's fine,
because the fix belongs on my side.

Polars never promised that every branch sees every row. I assumed it. Third pin: every row. The plugin owns its rows
now.

**Forces next:** Every row runs now. Is the number on each row right?

---

### Beat 5: Every row runs now. Is the number on each row right?

**Question:** Every row runs now. Is the number on each row right?

**Natural attempt:** test every method against scipy on the same inputs, plus property tests (hypothesis invents
thousands of inputs and checks what must always hold). And believe, reasonably, that the risky maths is the part you
cannot read: statrs' special functions. Your own algebra is the safe part.

**Wall:** scipy and polars-stats share 64-bit floats, so in the far tails both saturate and agree on the same wrong
`0.0`; parity cannot see past the shared limit. A 50-digit oracle (mpmath) found **eleven** defects where three were
budgeted, and most of them were in the elementary closed forms the speaker wrote, not in statrs.

**Move:** compute the quantity directly instead of as a difference of nearly equal numbers, standardise last, and work
in logarithms in the tails. Where the limit is statrs' (a Beta cdf of `-1.147`), report it upstream with a reproducer
instead of patching it for one project, and document what is still open. A report is a contribution.

**Picture:**

1. The ruler: the scipy figure from beat 1, with "parity on every method" and "property tests" as two green ticks.
2. The belief, as a slide with a thought bubble over the stack: an arrow to statrs labelled "risky (I can't read
   it)", an arrow to the thin strip labelled "safe (I wrote it)".
3. A float number line: the ruler (scipy) and the speaker's code both stop at `1e-308`; a much longer ruler appears
   beside them: "mpmath, 50 digits".
4. The budget slide: "budgeted: 3", click: "found: 11". Click: the breakdown: "Bernoulli + Uniform: 4. Exponential:
   3. Nine of eleven: one-to-three-line fixes in closed forms I wrote."
5. The favourite, worked live on the slide:
   `Bernoulli(p=1e-300).sf(0.5)` should be `1e-300` · computed as `1 - (1 - p)` · in float64 `1 - 1e-300 == 1.0` ·
   result `0.0`.
6. The sensor frame gains `log_tail`: the two faults, `sf` both `0.0`, `log_sf` `-1805` and `-2817` (true tails
   around `1e-784` and `1e-1224`). Sorting on `log_tail` puts the worse fault first.
7. The statrs limit: `Beta(s, s).cdf(0.5)` should be exactly `0.5`. Three rows: `s = 1e6` gives `0.4912`,
   `s = 1e7` gives `0.2129`, `s = 1e8` gives `-1.147` (the last one revealed on its own click).
8. Upstream, as three columns of issue cards: **fixed and released** in statrs 0.19.1 (3: `Beta.sf` at tiny `x`,
   `Beta.pdf` at 0, `Beta.ln_pdf` near 1; one through the speaker's own PR, folded into a contributor's) ·
   **fixed and merged, awaiting release** (2: `Beta.cdf` at large shapes, `LogNormal.pdf` left tail, both fixed by
   other contributors) · **open** (5: Beta's inverse cdf in the extreme lower tail, `Binomial.entropy` at
   `n = u64::MAX`, three Cauchy tails). Footnote: "status as of 2026-10-03; open ones are documented in polars-stats'
   accuracy page". The strip gets its fourth pin: **digits**. The thought-bubble arrows from slide 2 swap places.

**Terms introduced here:** 64-bit float (float64), underflow, hypothesis (property-based testing), mpmath, oracle,
catastrophic cancellation (shown, then named), `log_sf`.

**Say:**

Every row runs. Is the number on each row right?

The natural answer: check against scipy. [click] Every method is tested against scipy on the same inputs, plus
property tests: a tool called hypothesis invents thousands of inputs and checks what must always hold, like a cdf
staying between zero and one and never going down.

All green. [click] At that point I believed the risky code was the code I couldn't read, the special functions inside
statrs, because those are the hard parts. The algebra I'd written myself, one minus this, divided by that, was the safe
part. I could read it.

But scipy and polars-stats both compute in 64-bit floating point. [pause] In the far tails both run out of digits, and
two zeros agree perfectly. Parity can't tell you who's right where everybody saturates.

[click] So I brought a longer ruler: mpmath, a Python library that computes with as many digits as you ask for. Fifty,
here, across every method and distribution, with parameters many orders of magnitude apart. I'd budgeted time for three
defects.

[click] It found eleven. [pause] And not where I expected. [click] Bernoulli and Uniform, the two trivial
distributions, produced four of them. Exponential produced three more. Nine of the eleven were one-to-three-line fixes,
in closed forms I'd written myself.

[click] My favourite. A Bernoulli coin with p equal to ten to the minus three hundred. What's the probability of a value
above one half? It's p, a perfectly representable number. I returned zero. I computed it as one minus the cdf, and the
cdf is one minus p. In 64-bit floats, one minus ten to the minus three hundred is exactly one. The p was gone before
the subtraction even happened.

The fixes were small: compute what you want directly, not as the difference of two nearly equal numbers; standardise
last, not first; in the tails, work in logarithms. [click] Back to our sensors: the true tails of the two faults are
around ten to the minus 784 and ten to the minus 1224. `sf` says zero for both, so you can't rank them. `log_sf` says
minus 1805 and minus 2817, and the worse fault sorts first.

[click] And some limits really were in statrs. A Beta distribution with two equal shapes is symmetric, so its cdf at one
half is exactly one half. At shapes of a million, statrs 0.19 says 0.49. At a hundred million, it says [pause] minus
1.147. A probability below zero.

I could have patched that inside polars-stats and moved on. That fixes it for one project. Instead, each limit went
upstream as an issue with a reproducer: the input, the 50-digit answer, and what statrs returns. About ten reports.
[click] Three were fixed and released within days, in statrs 0.19.1, and one of those fixes folded in a small pull
request of mine. Two more were fixed by other contributors and are merged, waiting for the next release; that Beta cdf
is one of them. The rest are still open, including Beta's inverse cdf deep in the lower tail, which can panic and take
the whole query down with it. Until they're fixed, the polars-stats docs list each one, with the regime where it bites
and by how much.

I had underrated that part of open source. A good bug report is a contribution: it cost me an evening, and it fixed
the maths for everyone who uses statrs, not only for polars-stats.

Fourth pin: digits. [click] And the arrows swap. The most dangerous maths in polars-stats was the maths I wrote myself.

**Forces next:** Was it worth it, and what does it cost?

---

### Beat 6: Was it worth it, and what does it cost?

**Question:** Was it worth it, and what does it cost?

**Natural attempt:** show a speed chart. Rust plugin, so faster than scipy.

**Wall:** a speed chart without its conditions is marketing. One laptop, one engine with ten threads against a
library computing on one, and three ways of passing parameters that are different code paths on both sides.

**Move:** disclaimers first, then the two jobs from beat 1 (simulate: `sample`, `samples`; score: `cdf`, `sf`) at
10 thousand, 1 million and 10 million rows, read within one parameter regime at a time. Then the costs, the plan, and
the lesson on the opening picture.

**Picture:**

1. Disclaimers, a full slide, before any number (see "Benchmark conditions" below).
2. The three regimes as three small code lines: `Normal(0.0, 1.0)` (scalar) · `Normal(pl.col("mu"), pl.col("sigma"))`
   (column) · `Normal(pl.lit(0.0), pl.lit(1.0))` (broadcast). Caption: "never compare across them".
3. Results, column regime only, two dot plots side by side: **simulate** (`sample`, and `samples` with 10 draws per
   row) and **score** (`sf`). One row per distribution, three dots per row (10k, 1M, 10M rows), x axis the ratio
   scipy time / polars-stats time on a log scale (say "log" out loud), a vertical line at 1. Bernoulli and
   DiscreteUniform drawn in a muted colour with the footnote: "scipy: `randint` with per-row bounds runs
   `np.vectorize` (one Python call per row); `bernoulli` draws via NumPy `binomial(1, p)`". Headline numbers at 1M
   rows, excluding those two: `sample` 1.2x to 7.6x, `samples` 4.4x to 19x, `sf` 2.3x to 7.4x. Data: "Benchmark
   results" below.
4. Click: where scipy is ahead, not hidden. At 10k rows `sample` favours scipy in 11 of 12 column cells (up to 6.3x),
   12 of 12 scalar cells (up to 6.7x) and 12 of 12 broadcast cells (up to 25x): polars-stats costs 0.24 to 0.35 ms per
   call (0.5 to 0.9 ms with one-value expressions) whatever the distribution, against 0.04 to 0.17 ms for scipy. At
   10k, `cdf` and `sf` are roughly even (median ratios 0.86 to 0.99). With one-value expressions scipy also stays
   ahead on `sample` for DiscreteUniform at 1M and 10M (0.67x, 0.94x) and Uniform at 1M (0.89x).
5. What it costs, four short lines.
6. What is next: "every univariate distribution statrs supports".
7. The opening stack, now with four pins on the thin strip: order, length, every row, digits. The six-row frame next to
   it with all its columns.
8. The lesson, alone on a slide.
9. Credits and links.

**Terms introduced here:** benchmark regime (scalar, column, broadcast), frozen API, thread pool (recalled).

**Say:**

Was it worth it? Before any speed number, here's everything wrong with the measurement. [click]

One machine: my laptop. An Apple M5, ten cores, four performance and six efficiency, 24 gigabytes of memory, macOS,
plugged in, other applications closed, but still a laptop. Release builds, and every number is a median of repeated
timings. Polars 1.44.2 on the streaming engine, with ten threads. scipy 1.18.1 with NumPy, computing on one thread.
Both sides sample from a modern PCG generator. On the scipy side I measure the classic API, `norm(loc, scale).sf(x)`,
which is what most scipy code out there looks like; scipy's newer API is faster than that, 1.3 to 1.9 times on the
Normal when I measured it. And Polars pays for planning the query on every call, which scipy doesn't.

[click] There are three ways to pass parameters, and they're different code paths on both sides, so never compare
across them: plain numbers, one value per row, and a one-value expression that gets stretched.

[click] Here's the per-row case, the reason polars-stats exists. Each dot is a distribution, on a log scale; right of
the line, polars-stats is faster. At a million rows, one draw per row is 1.2 to 7.6 times faster than classic scipy,
ten draws per row 4 to 19 times, and scoring with `sf` 2.3 to 7.4 times. At ten million, a little more. Two dots sit
far to the right, at tens and hundreds of times: the discrete uniform, which scipy samples through a Python-level loop
when the bounds differ per row, and Bernoulli, which goes through NumPy's general binomial sampler. Don't quote those.

[click] At ten thousand rows, scipy wins nearly every one-draw-per-row cell, by up to six times. It draws ten
thousand numbers in about fifty microseconds. polars-stats spends about three hundred building and dispatching the
query, whatever the distribution. Large frames amortise that fixed cost; small frames pay it.

So these numbers are polars-stats and Polars' thread pool together, against classic scipy, on this laptop. That's
what you get when you use it, so that's what I measured.

[click] What it costs: a much shorter catalogue than scipy's hundred-plus, and no fitting, no multivariate
distributions, no statistical tests. Where you need those, scipy is the right tool, and the two live happily in one
project. And a compiled dependency, which has to follow Polars' plugin interface across releases.

[click] What's next: the short-term plan is to roll out every univariate distribution that statrs supports. The glue is
in place and tested on both engines, so each new one is mostly a file of one-line shells and a row in a test registry.

[click] Back to the stack. I wrote none of Polars' engine and none of statrs' maths, and scipy gave me the vocabulary
and the ruler. polars-stats, the thin strip in the middle, carries every pin: order, length, every row, digits. Every
bug that mattered lived there, in a promise nobody had made me.

[click] So if you take one thing home: borrow the engine, borrow the maths, and enforce what neither of them promises
you inside your own code, at the first line of the function, before it becomes a bug. And when the bug is in statrs
or Polars, report it upstream with a reproducer.

[click] Thank you to scipy, to Polars and to statrs, and to the people behind pyo3-polars, rand, mpmath and hypothesis.
Thank you to the statrs contributors who turned my reports into fixes, to sidsri14, who contributed two distributions
to polars-stats, and to camriddell, who fixed its docs. What I vouch for is the behaviour, and the tests that pin it.
It's `pip install polars-stats`, and the docs are on the slide. Thank you.

**Forces next:** none. Q&A.

## Takeaways (max 3)

1. **Borrow the engine, borrow the maths, and enforce what neither promises you inside the plugin.** Order, length,
   every row, digits: each was a promise nobody made.
2. **A plugin sees pieces, not your query.** Key anything stateful (a seed) on row identity, and align inputs by
   their length at run time.
3. **Your own algebra is maths too, and a report is a contribution.** Check both against an oracle with more digits
   than either library, fix what is yours, and send what is the giant's upstream with a reproducer.

## Benchmark conditions (the disclaimer slide, verbatim facts)

* Machine: Apple M5, 10 cores (4 performance, 6 efficiency), 24 GB, macOS 26.7.1. Local run, one machine, on AC power,
  other apps closed. Each run started only after three consecutive readings of the 1-minute load average below 2.5
  (2.05 at the start); the sweep's own ten Polars threads then hold it around 4 to 6. Run on 2026-10-03, 11:17 to
  11:49. Pareto and Weibull were re-run at 11:51 on a quiet machine, because short verification scripts overlapped
  their first run (load 12); the tables use the re-run.
* Software: Python 3.12.13, Polars 1.44.2, scipy 1.18.1, NumPy 2.5.3, polars-stats 0.1.0 built from `main` at
  `3d96178` in release mode (the harness refuses a debug build).
* Polars side: streaming engine, `collect(engine="streaming")`, default thread pool (10 threads). The timed call
  includes query construction and dispatch.
* scipy side: classic frozen API (`norm(loc=..., scale=...)` then the method). Samplers receive a
  `numpy.random.Generator` (PCG64), built per call. scipy computes these methods on one thread.
* Methods: `sample` (one draw per row), `samples` (10 draws per row), `cdf`, `sf`. Every distribution in the
  harness registry.
* Rows: 10,000 · 1,000,000 · 10,000,000. Regimes: scalar, column, broadcast; ratios are read within one regime only.
* Timing: median of up to 50 iterations within a 10-second budget per cell (minimum 3), after one warmup. Ratio is
  scipy median divided by polars-stats median (above 1 means polars-stats is faster).
* Not measured: scipy's newer distribution API (measured separately earlier on the Normal, column parameters: 1.3 to
  1.9 times faster than the frozen API on value-keyed methods); peak memory.
* In the column regime scipy draws `samples` as `(draws, rows)` and transposes a view, so it never pays for the
  row-major layout the Polars side materialises: that ratio is optimistic for scipy by about one pass over the output.

Reproduce:

```bash
make install-release
uv run --group tools -m tools.benchmarks.run --methods sample samples cdf sf \
    --rows 10_000 1_000_000 10_000_000 --max-seconds 10 --format json
```

## Benchmark results

Every cell passed the harness's correctness gate (shape for samplers, `allclose` for `cdf` / `sf`). Medians in ms are in
the raw JSON; the tables give the ratio only.

How to read them:

* **Column regime is the one to quote**: per-row parameters are why polars-stats exists. Excluding the two outliers,
  at 1M rows `sample` is 1.2x (Uniform) to 7.6x (Binomial), `samples` 4.4x (Normal) to 19x (Binomial), `cdf` 2.3x
  (Binomial) to 6.4x (Cauchy), `sf` 2.3x (Binomial) to 7.4x (Exponential). At 10M rows: `sample` 1.6x to 8.4x,
  `samples` 5.5x to 19x, `cdf` 2.6x to 9.6x, `sf` 2.5x to 9.2x.
* **The outliers have a known scipy-side mechanism**: `randint` with per-row bounds runs `np.vectorize` over
  `rng_integers` (one Python call per row), and `bernoulli` samples through NumPy's `binomial(1, p)`. Do not quote
  them as headline numbers.
* **At 10k rows scipy is ahead on `sample`** in 35 of 36 cells (and on `samples` in 16 of 36), because polars-stats
  pays a near-constant 0.24 to 0.35 ms per call (0.5 to 0.9 ms in the broadcast regime) for building and dispatching
  the query.
* **The closest large-frame cells**: Binomial `cdf` / `sf` stays around 2x at 1M and 10M in every regime, and Beta's
  does too with constant parameters (1.8x to 2.6x).
* **Raw data** is machine-specific and not kept: these tables are the curated record, and the command above regenerates
  the JSON.

### Column regime (one parameter value per row)

Ratio = scipy median / polars-stats median; above 1 means polars-stats is faster.

| Distribution | sample 10k | 1M | 10M | samples 10k | 1M | 10M | cdf 10k | 1M | 10M | sf 10k | 1M | 10M |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Normal | 0.17 | 2.64 | 2.49 | 0.66 | 4.42 | 6.18 | 0.93 | 4.51 | 5.00 | 0.87 | 4.66 | 6.20 |
| LogNormal | 0.26 | 2.36 | 2.74 | 1.29 | 5.06 | 5.49 | 0.96 | 3.76 | 4.15 | 1.10 | 3.81 | 4.20 |
| Cauchy | 0.42 | 3.50 | 3.75 | 1.96 | 7.33 | 8.21 | 0.92 | 6.42 | 6.74 | 0.86 | 5.99 | 6.83 |
| Pareto | 0.31 | 3.39 | 3.16 | 1.38 | 5.84 | 6.02 | 0.75 | 2.92 | 3.32 | 0.79 | 4.12 | 4.76 |
| Weibull | 0.42 | 3.40 | 3.13 | 1.97 | 5.63 | 5.48 | 0.95 | 2.83 | 3.41 | 0.80 | 2.63 | 2.96 |
| Uniform | 0.16 | 1.24 | 1.56 | 0.67 | 4.86 | 5.67 | 0.47 | 2.96 | 3.08 | 0.47 | 3.09 | 3.23 |
| Exponential | 0.18 | 3.22 | 2.86 | 0.73 | 5.65 | 6.27 | 0.83 | 5.95 | 6.43 | 0.58 | 7.41 | 8.55 |
| Beta | 0.65 | 3.74 | 4.75 | 2.45 | 5.88 | 7.45 | 3.45 | 6.23 | 9.62 | 4.44 | 6.75 | 9.17 |
| Bernoulli | 0.65 | 16.37 | 18.38 | 3.67 | 51.67 | 64.68 | 0.90 | 17.82 | 26.90 | 0.79 | 14.07 | 21.01 |
| Binomial | 0.98 | 7.55 | 8.37 | 7.01 | 18.81 | 18.90 | 1.32 | 2.26 | 2.60 | 1.31 | 2.26 | 2.52 |
| DiscreteUniform | 17.77 | 305.71 | 288.82 | 96.38 | 699.84 | 772.45 | 1.25 | 7.54 | 9.29 | 1.24 | 7.02 | 9.19 |
| Geometric | 0.32 | 4.86 | 5.49 | 2.26 | 12.32 | 12.72 | 0.92 | 4.06 | 4.93 | 0.85 | 6.17 | 7.27 |

### Scalar regime (Python numbers)

| Distribution | sample 10k | 1M | 10M | samples 10k | 1M | 10M | cdf 10k | 1M | 10M | sf 10k | 1M | 10M |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Normal | 0.18 | 2.37 | 2.86 | 0.73 | 6.33 | 7.50 | 1.00 | 5.87 | 6.63 | 0.95 | 5.76 | 6.68 |
| LogNormal | 0.26 | 2.62 | 3.04 | 1.37 | 5.36 | 5.69 | 1.14 | 4.84 | 5.19 | 1.17 | 5.03 | 5.37 |
| Cauchy | 0.47 | 4.74 | 4.97 | 1.95 | 8.14 | 8.79 | 0.95 | 8.06 | 8.62 | 0.92 | 8.03 | 8.60 |
| Pareto | 0.34 | 3.59 | 4.50 | 1.41 | 6.44 | 6.73 | 0.89 | 3.91 | 4.54 | 0.91 | 5.62 | 6.95 |
| Weibull | 0.45 | 3.70 | 4.41 | 1.80 | 6.17 | 6.58 | 1.07 | 3.85 | 4.77 | 0.74 | 2.74 | 3.57 |
| Uniform | 0.17 | 2.47 | 3.49 | 0.63 | 7.28 | 9.11 | 0.41 | 7.72 | 11.72 | 0.55 | 9.16 | 14.57 |
| Exponential | 0.18 | 2.38 | 2.94 | 0.74 | 6.38 | 7.25 | 0.92 | 7.49 | 8.86 | 0.76 | 10.63 | 14.11 |
| Beta | 0.61 | 4.96 | 5.56 | 2.30 | 7.01 | 7.35 | 1.06 | 2.04 | 2.31 | 1.12 | 2.18 | 2.55 |
| Bernoulli | 0.27 | 1.94 | 7.17 | 1.32 | 16.29 | 18.79 | 0.96 | 23.57 | 39.50 | 0.84 | 18.75 | 32.13 |
| Binomial | 0.52 | 5.10 | 5.20 | 2.42 | 7.96 | 8.23 | 1.06 | 1.89 | 2.07 | 1.03 | 1.80 | 2.00 |
| DiscreteUniform | 0.15 | 1.40 | 1.80 | 0.43 | 3.05 | 3.42 | 0.98 | 21.63 | 29.50 | 0.97 | 18.80 | 27.17 |
| Geometric | 0.28 | 3.22 | 3.67 | 1.16 | 6.27 | 6.85 | 1.06 | 8.94 | 10.46 | 1.14 | 16.24 | 21.61 |

### Broadcast regime (one-value expressions such as `pl.lit(0.0)`)

| Distribution | sample 10k | 1M | 10M | samples 10k | 1M | 10M | cdf 10k | 1M | 10M | sf 10k | 1M | 10M |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Normal | 0.05 | 1.24 | 1.87 | 0.35 | 4.71 | 8.23 | 0.56 | 4.43 | 4.64 | 0.94 | 6.15 | 6.41 |
| LogNormal | 0.09 | 1.76 | 2.45 | 0.68 | 4.64 | 5.12 | 1.10 | 4.84 | 5.18 | 1.30 | 5.16 | 5.35 |
| Cauchy | 0.14 | 2.62 | 3.47 | 0.97 | 7.11 | 7.56 | 0.93 | 8.01 | 8.57 | 0.88 | 7.92 | 8.28 |
| Pareto | 0.11 | 2.10 | 2.80 | 0.72 | 5.23 | 5.86 | 0.89 | 4.08 | 4.39 | 0.93 | 5.64 | 6.73 |
| Weibull | 0.15 | 2.49 | 3.31 | 1.06 | 5.78 | 5.44 | 0.88 | 3.53 | 3.94 | 0.85 | 3.63 | 3.79 |
| Uniform | 0.05 | 0.89 | 1.20 | 0.31 | 3.94 | 5.01 | 0.54 | 9.62 | 12.95 | 0.55 | 9.41 | 14.25 |
| Exponential | 0.09 | 1.72 | 2.45 | 0.45 | 5.29 | 5.48 | 0.95 | 3.81 | 8.85 | 0.72 | 10.81 | 14.52 |
| Beta | 0.20 | 3.24 | 4.10 | 1.33 | 6.40 | 7.36 | 0.95 | 1.84 | 2.13 | 0.99 | 2.07 | 2.46 |
| Bernoulli | 0.14 | 3.72 | 5.09 | 0.86 | 13.90 | 16.30 | 1.27 | 17.82 | 41.30 | 1.12 | 24.43 | 33.73 |
| Binomial | 0.15 | 2.43 | 2.90 | 1.27 | 5.89 | 6.57 | 1.02 | 1.92 | 2.15 | 0.96 | 1.87 | 2.05 |
| DiscreteUniform | 0.04 | 0.67 | 0.94 | 0.23 | 2.04 | 2.58 | 0.88 | 21.19 | 29.91 | 0.85 | 18.46 | 26.47 |
| Geometric | 0.13 | 1.85 | 2.34 | 0.73 | 5.43 | 5.80 | 1.04 | 8.54 | 10.37 | 1.14 | 17.52 | 20.98 |

## Links

* Docs: <https://fbruzzesi.github.io/polars-stats/>
* Code: <https://github.com/FBruzzesi/polars-stats> · `pip install polars-stats`
* scipy.stats: <https://docs.scipy.org/doc/scipy/reference/stats.html>
* Polars expression plugins: <https://docs.pola.rs/user-guide/plugins/>
* statrs: <https://docs.rs/statrs>
* pyo3-polars: <https://github.com/pola-rs/pyo3-polars>
* mpmath: <https://mpmath.org>

## Cut list

Cut whole asides, never a wall or a prediction. In this order:

1. Beat 2, the macros paragraph (20 s). Keep the one-line shell on the slide.
2. Beat 3, the Binomial aside (20 s). Keep the slide as a footnote: "statrs Binomial sampling is O(n); `rand_distr`".
3. Beat 1, the scipy vocabulary list shrinks to "`pdf`, `cdf`, `sf`, and friends" with the full list on the slide
   (20 s).
4. Beat 5, the upstream status shrinks to one sentence: "Three fixed and released within days, two merged, the rest
   open and documented." Keep the `-1.147` and the "a report is a contribution" line.
5. Beat 6, the costs paragraph shrinks to: "Shorter catalogue, no fit, a compiled dependency."

Never cut: the `nan` quiz (beat 1), the `explain()` plan (beat 2), the two-engine prediction and the alternatives table
(beat 3), the one-versus-ten prediction and the `0.0` on Polars 1.44 (beat 4), three-versus-eleven, the `1 - (1 - p)`
slide and the upstream story (beat 5), the disclaimers (beat 6).

## Sources

Every number and incident above, with where it comes from. "Re-run" means reproduced on 2026-10-03 on the machine
above.

| Claim | Source |
|---|---|
| `stats.norm(loc=0, scale=-1).sf(1.0)` is `nan`, no warning | Re-run, scipy 1.18.1, warnings captured: none |
| Polars has no `erf` | Re-run, Polars 1.44.2: no attribute containing `erf` on `pl.Expr` |
| `explain()` output, filter inside the Parquet scan; 482 rows identical on both engines | Re-run, Polars 1.44.2, polars-stats `main`, 1,000,000-row file of the two tutorial sensors |
| First commit: 26 April 2026, `Bernoulli.sample`, one `ChaCha20Rng` and a row loop, under seventy lines of Rust, the "not supported" docstring and the `pl.repeat` comment | PR #1 (`e0d4a16`); comment and docstring quoted verbatim |
| Single seeded stream made output depend on chunk boundaries; fixed by `(seed, index)` seeding before any release | PR #6 (2026-05-26); first PyPI release was 0.0.1 on 2026-08-09 |
| Per-row `ChaCha20` 10 to 20 times slower; `Pcg64Mcg` took 1M × 100 from about 5.9 s to 0.29 s | PR #7 (2026-06-01) |
| statrs samples Binomial in O(n); `rand_distr` instead | PR #28 (2026-06-11) |
| `samples` as one native call, `samples(size=1)` equals `sample` | PR #30 (2026-06-14) |
| CI runs every test on both engines | PR #57 (2026-08-21) |
| Four macros replaced by generic functions; plugins are one-line shells | PRs #49 to #56 (2026-08-21), #94 (2026-09-12) |
| 0.0.1, `sigma=pl.lit(0.5)`, 1M rows: 1 distinct value in-memory, 10 streaming, height 1,000,000, no error; 0.0.2 correct | Re-run from PyPI wheels in a clean environment outside the checkout, Polars 1.43.2 |
| Alignment moved to Rust, breaking release 0.0.2 | PR #58 (2026-08-24); 0.0.2 released 2026-08-30 |
| `over` example (per-sensor `mean` / `std` as parameters): `ShapeError` on 0.0.1 in-memory, correct on `main` on both engines | Re-run: 0.0.1 from PyPI on Polars 1.43.2; `main` on Polars 1.44.2, agreeing with scipy within the parity suite's `1e-10` relative tolerance. Not claimed bit-identical across engines: Polars' own `mean` / `std` differ by a few ulps between engines |
| Polars 1.44 masks unselected `when / then` arms | Upstream Polars PR #28498; issue filed as Polars #29005 (open) |
| 0.0.1 (requires `polars>=1.15.0`, no cap): Uniform with `max < min` raises on Polars 1.43.2, returns `0.0` on 1.44.2 | Re-run from PyPI wheels, plain `pip install` resolution |
| About ten days from cap to wrapper deletion | PR #75 (cap, 2026-08-29), #79 to #83 (port, lift), #88 (wrapper deleted, 2026-09-08) |
| Parity against scipy plus hypothesis invariants | PR #10 (2026-06-02) |
| mpmath at 50 digits; 11 defects against 3 budgeted; Bernoulli and Uniform 4, Exponential 3; nine one-to-three-line closed-form fixes; `Bernoulli(1e-300).sf(0.5)` was `0.0` | PR #43 (2026-08-09) and the maintainer's audit notes of 2026-08-07 (the count is not in a public PR body) |
| Sensor faults: `log_sf` `-1805` and `-2817`, true tails about `1e-784` and `1e-1224` | Docs tutorial, step 5 (executed at docs build) |
| polars_rng: no `seed` argument, thread-local generator | Its README and the comparison in the polars-stats docs (`docs/index.md`, Related projects); not installed here (no macOS arm64 wheel) |
| polars-random 0.5.0, `seed=42`, 1M rows: re-run same; in-memory vs streaming different (scalar and column parameters); 1 vs 4 chunks different with column parameters; polars-stats same in all three | Re-run in a clean environment, Polars 1.44.2 |
| `Beta(s, s).cdf(0.5)`: `0.4912`, `0.2129`, `-1.147` at `s = 1e6, 1e7, 1e8` | Re-run on `main` (statrs 0.19.1); also in the accuracy page |
| statrs reports (filed 2026-08-07 to 2026-09-18): fixed and released in 0.19.1 (2026-08-11): #432, #433, #437 via statrs PR #438 (the speaker's PR #436 was merged into it); fixed and merged after 0.19.1, unreleased: #434 (PR #456), #452 (PR #454); open: #435, #449, #471, #472, #473 | GitHub, checked 2026-10-03. Re-check the week before the talk: counts will move |
| Benchmark ratios, outlier mechanism | The sweep above; scipy 1.18.1 source of `randint_gen._rvs` (`np.vectorize`) and `bernoulli_gen._rvs` (`binom_gen._rvs` with `n = 1`) |
| scipy newer API 1.3 to 1.9 times faster than the frozen API on Normal, column parameters | Benchmark harness README, "Keeping the scipy side native" |
| Contributors | polars-stats: sidsri14 (PRs #64, #68), camriddell (PR #110). statrs fixes for these reports: day01 (PR #438, #456), teddytennant (PR #454), merged by the maintainer YeungOnion; names for the credits slide if you want them there |
