# Vectorized statistical distributions, built on the Polars engine

**Destination (one sentence):** Three giants hand you the vocabulary (scipy), the engine (Polars) and the maths
(statrs), and every bug that mattered lived in the thin layer between the engine and the maths, in the promises neither
of them ever made: the order rows arrive in, how long each input is, which rows actually run, and how many digits
survive.

**The lesson, as one instruction:** borrow the engine, borrow the maths, and enforce what neither promises you inside
the plugin, before it becomes a bug.

**The three giants, and what each gives.** Each is introduced at the moment the story needs it, and credited there for
what it gives, not only thanked at the end:

| Giant | What it is | What it gives polars-stats | Introduced |
|---|---|---|---|
| scipy | Python's scientific library, since 2001; `scipy.stats` has more than a hundred distributions | The vocabulary (every method name, so porting scipy code is mostly renaming) and the ruler (every method of every distribution is tested against it) | Beat 1 |
| Polars | A dataframe query engine written in Rust, driven mostly from Python | The engine (lazy planning, pushdown, threads, streaming, groups) and the door into it (expression plugins, which the Polars team ships and documents) | Beats 1 and 2 |
| statrs | The closest thing Rust has to `scipy.stats`, maintained by volunteers | The maths: per distribution, the density, cdf, survival function, inverse cdf, log density and sampling, with `erf`, gamma and beta underneath | Beat 2 |

**Smallest concrete case:** two temperature sensors from the docs tutorial. `temp-a` sits around 10 degrees
(`mu = 10`, `sigma = 0.5`), `temp-b` around 100 (`mu = 100`, `sigma = 2`). Six readings, two of them faults (`40.0` on
`temp-a`, `250.0` on `temp-b`). Two jobs, which were my own first use case: **simulate** readings per sensor
(`sample`, `samples`) and **score** real ones against their own sensor's baseline (z-score, then tail probability,
`sf`). Where it stops transferring: real telemetry has drift and non-normal noise; the mechanics do not care.

**Visual spine: the stack.** Polars on top (the engine), statrs at the bottom (the maths), and between them a thin
strip labelled "polars-stats". scipy, the third giant, stands beside the stack as "the vocabulary, and the ruler": it is
not in the running code, it defines what the code should answer. Each beat pins one promise onto the thin strip:
*order*, *length*, *every row*, *digits*. On the right, the six-row sensor frame is the running example and gains one
column per beat (`z`, `upper_tail`, `simulated`, `log_tail`). The closing slide is the opening stack, now carrying all
four pins.

**Audience / assumed knowledge / slot:** compute! Paris 2026, people who work with open-source computation and data,
across languages. Assume they know what a table, a function and a probability are. Assume nothing about Python, NumPy,
scipy, Polars, Rust or statrs: each is introduced at the moment the story needs it, and every term is defined the first
time it is spoken (each beat's "Terms introduced" line is the checklist). 25 minutes plus 5 of Q&A. No live demo: every
output on a slide is a recorded, re-run result (see Sources).

**Promise (said in the first minute):** after this talk you can build on someone else's engine and someone else's
maths without being an expert in either, and you will know where your own bugs are going to live.

**House rules for this script:** never a count of the distributions shipped (it will change before November); sensor
telemetry only; neutral tone towards scipy (state what each returns and why, never "more correct"), and scipy is never
the villain: each wall it hits is a property of arrays, of float64 or of one thread, said as such; benchmarks only after
their disclaimers.

## Question chain

1. Each sensor has its own baseline. How do I simulate readings and score real ones, row by row, on a dataframe?
2. Can the distribution live inside the Polars query instead?
3. If the engine and the maths are borrowed, what is left for me to write?
4. The plugin sees pieces. Do the inputs arrive at full length, and does every row run?
5. Every row runs now. Is the number on each row right?
6. Was it worth it, and what does it cost?

Retelling test (what an attendee says at lunch): "The speaker built a stats library on Polars and a Rust maths crate,
with scipy as the blueprint and the ruler, without really knowing Rust. The engine and the maths came for free; every
bug was in the little bit of glue in between, in things nobody had promised."

## Timing

| Beat | Question (short) | Minutes | Ends at |
|---|---|---|---|
| 1 | How do I simulate and score per row? | 4.25 | 4:15 |
| 2 | Can it live inside the query? | 5.0 | 9:15 |
| 3 | What is left for me to write? | 3.75 | 13:00 |
| 4 | Full length, every row? | 4.25 | 17:15 |
| 5 | Is the number right? | 3.75 | 21:00 |
| 6 | Worth it, and the cost? | <TODO: after the benchmark rerun> | |

<TODO: word count and runtime after the benchmark rerun fills beat 6.> Rehearse with a timer: if beat 2 ends past 9:45,
take cut list items 1 to 3.

## Beats

### Beat 1: Each sensor has its own baseline. How do I simulate readings and score real ones, row by row, on a dataframe?

**Question:** the one the room already has: how do I do statistics on data that lives in a dataframe?

**Natural attempt:** leave the dataframe. Take the columns to NumPy (Python's array library), call `scipy.stats`, put
the results back. And it is a good attempt: scipy broadcasts parameter arrays, so a different distribution per row is
already vectorised.

**Wall:** the cost is where the result lands. The lazy query has to end before you can leave it, keeping the result
attached to the right rows is now your job, and an invalid parameter comes back as `nan` with no warning.

**Move:** want the result to stay an expression inside the query, the way R users have always had it with `pnorm`
inside `mutate`. (The how is beat 2.)

**Picture:**

1. The six-row frame, `sensor` and `reading` only. Question on the slide: "Two of these are faults. Which?"
2. Click: the baselines appear as `mu` and `sigma` columns, and a `z` column fills in (`z = (reading - mu) / sigma`).
3. Click: a bell curve per sensor, with the tail beyond each reading shaded. Label: "how far" (z) vs "how surprising"
   (tail probability).
4. scipy's slide: its vocabulary as a table (`pdf`, `cdf`, `sf`, `ppf` · `isf`, `rvs`, one plain-words gloss each),
   then the round trip as code:

    ```python
    df = lf.collect()  # the lazy query ends here
    mu, sigma, x = (df[c].to_numpy() for c in ("mu", "sigma", "reading"))
    rng = np.random.default_rng(42)
    df = df.with_columns(
        simulated=pl.Series(stats.norm(loc=mu, scale=sigma).rvs(random_state=rng)),
        upper_tail=pl.Series(stats.norm(loc=mu, scale=sigma).sf(x)),
    )
    ```

5. Click: three labels appear next to the code: "the query ends", "alignment is yours", "?".
6. Quiz slide: `stats.norm(loc=0, scale=-1).sf(1.0)` returns what? Click: `nan`. Footnote: no warning emitted.
7. Title: "R keeps the answer in the data frame. I wanted that in Polars". The R line,
   `mutate(df, upper_tail = pnorm(reading, mu, sigma, lower.tail = FALSE))`, above the frame with an empty
   `upper_tail` column; the stack appears for the first time, nearly empty: only scipy, standing at the side, labelled
   "the vocabulary, and the ruler".

**Terms introduced here:** z-score, tail probability, Polars (one sentence), lazy query, NumPy, scipy, `scipy.stats`,
`sf`, `rvs` (the other method names are on the slide only), optimiser, R's `pnorm` and `mutate`.

**Say:**

Here are six temperature readings from two sensors. Sensor A usually sits around 10 degrees, give or take half a
degree. Sensor B sits around 100, give or take 2. Two of these readings are faults. Which ones? [pause]

Forty on sensor A, two hundred and fifty on sensor B. You just did that in your head with a z-score: how many standard
deviations a reading sits from its sensor's mean. Forty is sixty standard deviations above ten.

[click] That tells you how far. What I usually need is how surprising: if the sensor is healthy, how likely is a
reading at least this high? That's the tail of the distribution, and it lets you set an alarm at "one in a thousand"
instead of at a magic number of degrees, different for every sensor.

And before I score real readings, I want fake ones: simulated readings for each sensor, from its own baseline, one
draw per row or a thousand per row. So, two jobs: sample and score. And every row has its own distribution.

I'm Francesco. One thing that matters for the next 25 minutes: I'm not a Rust expert, and a good part of the Rust in
polars-stats, the library this talk is about, was written with AI assistance. This talk is about what you can build
anyway, standing on three giants, and by the end you'll know where your own bugs are going to live.

My data lives in Polars, a dataframe engine written in Rust that most people drive from Python. Its trick is laziness:
you describe the whole query first, and Polars plans it before running any of it.

The first giant is scipy, Python's scientific library, around since 2001. Its statistics module, `scipy.stats`, has
more than a hundred distributions, and they all speak one vocabulary. `sf`, the survival function, is the tail I want.
`rvs` draws samples. The rest are on the slide. polars-stats copies that vocabulary word for word, so porting scipy code
is mostly renaming.

And scipy already solves my problem. Give it an array of means and an array of standard deviations, and it scores every
element against its own distribution, in compiled code, with no Python loop. So the honest first attempt is this one.
[click] Take the columns out as NumPy arrays, Python's standard arrays, give them to scipy, put the results back. It
works. I did it for years.

What it costs is where the result lands. [click] First, the query: to leave a lazy query you have to run it, so
everything after this line is a second query, and the optimiser, the part of Polars that rewrites your query before it
runs, never sees the two together. Second, alignment: the result is a bare array, and keeping it on the right rows
through the next join is your job. Third, a quiz. [click] A normal distribution with a standard deviation of minus one.
That's not a distribution. What does scipy give you for its tail? [pause]

[click] `nan`. No warning. A modelling error, travelling down your pipeline dressed up as missing data.

None of this is scipy's fault: an array has no query plan, no row identity and no null, so `nan` is the only way it can
say "invalid". [click] And if you come from R, you've been smiling for a minute: R's `pnorm`, inside `mutate`, has
always kept the answer in the data frame. That's what I wanted in Polars: the answer as one more expression in the
query.

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
engine side, and statrs for the maths. One flag, `is_elementwise=True`, is a promise, and the promise is what the
optimiser needs: flip it off and the optimisation goes away.

**Picture:**

1. The frame with `z` computed in plain Polars: `(pl.col("reading") - pl.col("mu")) / pl.col("sigma")`. Next to it,
   the formula for the tail, with `erf` in red: "Polars: no `erf`".
2. Click: the three attempts listed, each with its one-line cost.
3. Click: the stack fills in. Polars box on top ("the engine"), statrs box at the bottom ("the maths"), and between
   them a thin, empty strip labelled "polars-stats".
4. The anatomy, three layers top to bottom, one line each:

    ```python
    baseline = ps.Normal(mu="mu", sigma="sigma")  # Python: returns pl.Expr, computes nothing
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
7. Prediction slide: the same query with the plugin registered as `is_elementwise=False`. "Does the filter still
   move?" Click: its plan, side by side with the first one (sensor filter only, Polars 1.44.2):

    ```text
    is_elementwise=True                          is_elementwise=False
    WITH_COLUMNS: [… normal_sf …]                FILTER col("sensor") == "temp-b"
      Parquet SCAN [readings.parquet]            FROM
      SELECTION: col("sensor") == "temp-b"         WITH_COLUMNS: [… normal_sf …]
                                                     Parquet SCAN [readings.parquet]
    ```

8. Click: `collect(engine="in-memory")` and `collect(engine="streaming")`: 482 rows each, `equals` is `True`. Then
   one line each for `sink_parquet`, `over` / `group_by`, and "all 10 cores".
9. Back to the stack: Polars and statrs labelled with what each gave; the strip between them still empty.

**Terms introduced here:** `erf`, `map_elements`, expression plugin, Rust crate, statrs, `is_elementwise`, Parquet,
predicate pushdown (shown, then named), streaming engine, morsel, `group_by`, `over` (window function), thread pool.

**Say:**

Polars can do the first half on its own. [click] Reading minus mu, divided by sigma: plain arithmetic, on every core.
The next step is the problem. Turning a z-score into a tail probability needs a special function, the error function,
`erf`, and Polars doesn't have one. Fair enough: it's a dataframe engine, not a statistics package.

So what would you try? [pause] [click] Call a Python function once per row with `map_elements`: it works, one row at a
time, through the interpreter, on one thread. Type a textbook approximation of `erf` as Polars arithmetic: fine in the
middle, wrong in the tails, which is exactly where anomaly detection lives.

The third option is the one Polars offers itself: an expression plugin. You write a function in Rust, compile it, and
Polars calls it as if it were one of its own expressions. The Polars team ships the machinery for this, and documents
it. The function receives columns and returns a column, and that is the whole contract.

That leaves the maths, and I was not going to write an error function. [click] Meet the second giant: statrs. It's a
Rust crate, which is Rust's word for a library, and it's the closest thing Rust has to `scipy.stats`. Build
`Normal::new(mu, sigma)` and you get the density, the cdf, the survival function, the inverse cdf, the log density and
sampling, with `erf`, gamma and beta underneath. It's maintained by volunteers, and it's the reason polars-stats
exists.

[click] So here's the whole anatomy. On top, a Python class whose parameters can be column names. Its methods compute
nothing: each one returns a Polars expression that says "call this Rust function on these columns". In the middle, the
Rust function: it gets the columns, walks the rows, builds a distribution per row. At the bottom, one line:
`dist.sf(v)`. That line is statrs.

Now the part I enjoy the most. [click] When I register the function with Polars, I pass one flag:
`is_elementwise=True`. It's a promise: row i of the output depends only on row i of the inputs. Watch what Polars does
with that promise.

[click] Here's a pipeline. Scan a Parquet file, a common columnar file format, holding a million readings. Add the
tail probability, keep sensor B, keep the readings rarer than one in a thousand. And here's what Polars plans to do,
read from the bottom up. [pause] The filter on sensor B, which I wrote after the scoring, has moved down into the file
reader. Half the rows are never decoded, so they never reach my function. The filter on the score stays on top, because
it needs the score. That's called predicate pushdown, and I wrote none of it. In the scipy round trip, that reordering
was your job, by hand.

Is that the promise, or would Polars do it anyway? Same query, one change: `is_elementwise=False`. Does the filter
still move? [pause] [click] No. It stays on top, and every row gets scored. If my function could look at its
neighbours, removing rows first might change the answers, so Polars won't risk it. The promise is what buys the
optimisation.

[click] Run it on the in-memory engine, and on the streaming engine, which reads the file in small batches called
morsels, so the data never has to fit in memory. Same 482 rows, identical. Swap `collect` for `sink_parquet` and it
streams from file to file. Put it inside a `group_by`, or an `over`, Polars' window function, like SQL's `OVER`, and it
works per group. All of it on Polars' thread pool, across every core.

I wrote a loop over rows. Polars made it lazy, optimised, parallel, streaming and groupable. statrs did the maths.
scipy told me what right looks like.

[click] So the picture is this: Polars on top, statrs at the bottom, scipy beside them holding the ruler. And between
Polars and statrs, a thin strip of code that I actually wrote. That strip is polars-stats.

**Forces next:** If the engine and the maths are borrowed, what is left for me to write?

---

### Beat 3: If the engine and the maths are borrowed, what is left for me to write?

**Question:** If the engine and the maths are borrowed, what is left for me to write?

**Natural attempt:** the first commit, and the other job from beat 1: simulate. One distribution (`Bernoulli`, a
biased coin), one method (`sample`): seed one random number generator, walk the rows in order, draw one value per row
with statrs. Its docstring is honest about the limit: chunked and streaming inputs are "not supported".

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
8. The stack's thin strip gets its first pin: **order**. The frame gains `simulated` (`seed=42`, identical on both
   engines): `9.763219`, `10.825292`, `11.172469`, `98.532788`, `100.147242`, `97.717269`. Caption: "polars-random
   and polars_rng: good sampling plugins, and polars_rng's catalogue is wider. Neither gave a seeded column identical
   on both engines." The measured comparison is backup slide B1.

**Terms introduced here:** seed, random number generator, chunk, thread pool (recalled), morsel (recalled),
`ChaCha20`, `Pcg64Mcg`, `rand_distr`, `samples`.

**Say:**

The glue started very small, and with the other job from the first slide: simulating.

[click] 26 April 2026, the first commit of polars-stats. One distribution, Bernoulli: a biased coin that comes up 1
with probability p. One method, `sample`. Under seventy lines of Rust. The idea is what anyone would write: take the
seed, start one random number generator, walk the rows in order, and draw one value per row with statrs.

And the docstring was honest. [click] "Reproducibility under seed assumes a single-chunk input series; chunked or
streaming inputs are not supported."

Not supported. As if that were a mode the caller could switch off.

[click] Polars doesn't hand a plugin "the column". It hands it pieces. A column can be stored as several blocks, called
chunks. The thread pool splits work across cores. The streaming engine sends morsels. A group by sends one group at a
time. My function runs once per piece, and every call restarts the generator from the same seed.

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
sampling ten to twenty times slower. The fix for the fix was `Pcg64Mcg`, a small statistical generator that costs a
handful of integer operations to build. On that day's benchmark, a million rows times a hundred draws went from about
5.9 seconds to 0.29. And `samples`, a thousand draws per row, is the first thousand values of that row's own stream.

[click] One place where a giant said no: statrs draws a Binomial by flipping n coins, so n random numbers per draw.
Binomial sampling borrows from another crate instead, `rand_distr`, whose cost doesn't grow with n. Everything else
about the Binomial still comes from statrs.

[click] Today a seeded column repeats across runs, chunk layouts, thread counts and both engines, and CI runs every
test once per engine.

[click] So our frame gains its simulated column: one plausible reading per row, from that row's own sensor, the same on
every run. Two other Polars plugins sample well, polars-random and polars_rng, and polars_rng's catalogue is wider than
polars-stats'. Neither gave me a seeded column that stays identical across both engines, and that's the gap
polars-stats fills.

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
7. Polars 1.44, one line: "each branch is evaluated only on the rows that select it". Then the same published package
   (0.0.1), the same frame (`min = 1`, `max = 0`, `x = -5`), two Polars versions: 1.43.2 raises `ComputeError`, 1.44.2
   returns `0.0`.
8. The ten-day fix as four steps on a line: cap `polars<1.44` · move the closed forms into Rust · lift the cap ·
   delete the Python null wrapper. The strip gets two pins: **length**, **every row**.

**Terms introduced here:** broadcasting, `pl.lit`, PyPI, run time vs plan time, closed form, `when / then / otherwise`,
`ComputeError`, breaking release.

**Say:**

Here's another comment from that first commit. [click] Roughly: Polars doesn't broadcast inputs into a plugin, so a
one-value `pl.lit(p)` would draw a single sample that Polars then repeats across the frame. Broadcasting means
stretching one value to the length of the column. So Python stretched every plain number before calling Rust. Handled.

It handled the case I thought of. [click] But Polars also lets you write `pl.lit(0.5)`, a constant wrapped as an
expression, or a column's mean, or its first value. Each is one row long, and only the running engine knows it, not the
Python that builds the query. And the Rust helper that walks several columns together stops at the shortest one.

Prediction time. [click] Version 0.0.1, straight from PyPI, Python's package index. A million rows, a Normal with sigma
equal to `pl.lit(0.5)`, scored with `cdf`. The right answer has a million distinct values. How many do we get? [pause]

[click] In-memory: one. Row zero's answer, a million times. Streaming: ten, one per morsel. The right height, no
error, no warning. That one I published.

[click] The move: align in Rust, not in Python. The first thing every plugin does now is stretch any input of length one
to the others' length, and raise if two real lengths disagree, judged by actual length at run time, not by kind of
expression. It changed what some calls return, so it shipped as a breaking release, 0.0.2.

[click] That fix is what makes this line work. It estimates each sensor's baseline from its own readings and scores
every reading against it, in one expression, inside an `over`. The mean and the standard deviation are one value per
sensor: length one. The plugin stretches them, Polars does the grouping. On 0.0.1 this line raised a shape error on the
in-memory engine. Today it runs on both. Second pin: length.

Now, which rows run. Remember scipy's `nan` for a negative standard deviation? I wanted an error instead. But a Polars
expression can't raise on a bad row. Some methods have a closed form, a formula you can write in one line, so I wrote
those as Polars arithmetic, and sent the parameters through a tiny Rust plugin whose only job was to check and raise.
[click] And that check sat inside a `when / then / otherwise`, Polars' if-else, next to the null handling.

[click] Then Polars 1.44 shipped a good optimisation: evaluate each branch only on the rows that select it. For Polars'
own expressions, that's pure win. For my validator, it meant it no longer saw the rows it existed to reject.

Same published package, same three-row frame: a Uniform whose maximum is below its minimum. On Polars 1.43: an error,
as designed. On Polars 1.44: [pause] zero. A quiet number where an error should be. And a plain `pip install` would give
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
`0.0`; parity cannot see past the shared limit, which is float64's, not scipy's. A 50-digit oracle (mpmath) found
**eleven** defects where three were budgeted, and most of them were in the elementary closed forms the speaker wrote,
not in statrs.

**Move:** compute the quantity directly instead of as a difference of nearly equal numbers, and work in logarithms in
the tails. Where the limit is statrs' (a Beta cdf of `-1.147`), report it upstream with a reproducer instead of
patching it for one project, and document what is still open. A report is a contribution.

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
   result `0.0`. Last click names it: "catastrophic cancellation".
6. The sensor frame gains `log_tail`: the two faults, `sf` both `0.0`, `log_sf` `-1805` and `-2817` (true tails
   around `1e-784` and `1e-1224`). Sorting on `log_tail` puts the worse fault first.
7. The statrs limit: `Beta(s, s).cdf(0.5)` should be exactly `0.5`. Three rows: `s = 1e6` gives `0.4912`,
   `s = 1e7` gives `0.2129`, `s = 1e8` gives `-1.147` (the last one revealed on its own click).
8. Upstream, as three columns of issue cards, shown at once and not walked through: **fixed and released** in statrs
   0.19.1 (3: `Beta.sf` at tiny `x`, `Beta.pdf` at 0, `Beta.ln_pdf` near 1; one through the speaker's own PR, folded
   into a contributor's) · **fixed and merged, awaiting release** (2: `Beta.cdf` at large shapes, `LogNormal.pdf` left
   tail, both fixed by other contributors) · **open** (5: Beta's inverse cdf in the extreme lower tail,
   `Binomial.entropy` at `n = u64::MAX`, three Cauchy tails). Footnote: "status as of 2026-10-03; open ones are
   documented in polars-stats' accuracy page". The strip gets its fourth pin: **digits**. The thought-bubble arrows
   from slide 2 swap places.

**Terms introduced here:** 64-bit float (float64), significant digits, underflow, hypothesis (property-based testing),
mpmath, oracle, catastrophic cancellation (shown, then named), `log_sf`.

**Say:**

Every row runs. Is the number on each row right?

The natural answer: check against scipy. [click] Every method is tested against scipy on the same inputs, plus
property tests: a tool called hypothesis invents thousands of inputs and checks what must always hold, like a cdf
staying between zero and one and never going down.

All green. [click] At that point I believed the risky code was the code I couldn't read, the special functions inside
statrs, because those are the hard parts. The algebra I'd written myself, one minus this, divided by that, was the safe
part. I could read it.

But scipy and polars-stats both compute in 64-bit floating point, about sixteen significant digits. [pause] In the far
tails both run out of digits, and two zeros agree perfectly. That's not a scipy limit, it's a float64 limit: parity
can't tell you who's right where everybody saturates.

[click] So I brought a longer ruler: mpmath, a Python library that computes with as many digits as you ask for. Fifty,
here, across every method and distribution, with parameters many orders of magnitude apart. I'd budgeted time for three
defects.

[click] It found eleven. [pause] And not where I expected. [click] Bernoulli and Uniform, the two trivial
distributions, produced four of them. Exponential produced three more. Nine of the eleven were one-to-three-line fixes,
in closed forms I'd written myself.

[click] My favourite. A Bernoulli coin with p equal to ten to the minus three hundred. What's the probability of a value
above one half? It's p, a perfectly representable number. I returned zero. I computed it as one minus the cdf, and the
cdf is one minus p. In 64-bit floats, one minus ten to the minus three hundred is exactly one. The p was gone before
the subtraction even happened. Subtracting two nearly equal numbers has a name: catastrophic cancellation.

The fixes were small: compute what you want directly, not as a difference, and in the tails, work in logarithms.
[click] Back to our sensors: the true tails of the two faults are around ten to the minus 784 and ten to the minus
1224. `sf` says zero for both, so you can't rank them. `log_sf` says minus 1805 and minus 2817, and the worse fault
sorts first.

[click] And some limits really were in statrs. A Beta distribution with two equal shapes is symmetric, so its cdf at one
half is exactly one half. At shapes of a million, statrs 0.19 says 0.49. At a hundred million, it says [pause] minus
1.147. A probability below zero.

I could have patched that inside polars-stats and moved on. That fixes it for one project. Instead, each limit went
upstream as an issue with a reproducer: the input, the 50-digit answer, and what statrs returns. About ten reports.
[click] Three were fixed and released within days, two more are merged, that Beta cdf among them, and the rest are open
and listed in the polars-stats docs.

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

<TODO: rewrite from the benchmark rerun.>

**Terms introduced here:** benchmark regime (scalar, column, broadcast), frozen API, thread pool (recalled).

**Say:**

<TODO: rewrite from the benchmark rerun.>

**Forces next:** none. Q&A.

## Backup slides (after the thank-you, for Q&A only)

* **B1. The seeded-sampling comparison**, measured:

    | `seed=42`, 1M rows, Polars 1.44.2 | re-run | in-memory vs streaming | 1 chunk vs 4 chunks |
    |---|---|---|---|
    | polars_rng | no `seed` argument | | |
    | polars-random 0.5.0 | same | different | different (column parameters) |
    | polars-stats 0.1.0 | same | same | same |

    Caption: "polars-random and polars_rng are good sampling-first plugins, and polars_rng's sampling catalogue is
    wider than polars-stats'." Footnote: "measured 2026-10-03".
* **B2 to B4. The full benchmark tables**, one per regime (column, scalar, broadcast), from "Benchmark results" below.
* **B5. Benchmark conditions in full**, from "Benchmark conditions" below, including the reproduce command.

## Takeaways (max 3)

1. **Borrow the engine, borrow the maths, and enforce what neither promises you inside the plugin.** Order, length,
   every row, digits: each was a promise nobody made.
2. **A plugin sees pieces, not your query.** Key anything stateful (a seed) on row identity, and align inputs by
   their length at run time.
3. **Your own algebra is maths too, and a report is a contribution.** Check both against an oracle with more digits
   than either library, fix what is yours, and send what is the giant's upstream with a reproducer.

## Benchmark conditions (the disclaimer slide, verbatim facts)

<TODO: rewrite from the benchmark rerun.>

## Benchmark results

<TODO: rewrite from the benchmark rerun.>

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

1. Beat 3, the Binomial aside (20 s). Keep the slide as a footnote: "statrs Binomial sampling is O(n); `rand_distr`".
2. Beat 2, the `sink_parquet` / `group_by` / `over` sentence (15 s). Keep the line on the slide.
3. Beat 6, the costs paragraph shrinks to: "Shorter catalogue, no fit, a compiled dependency."
4. Beat 1, the R line (15 s). Keep the slide's click.

Never cut: the `nan` quiz (beat 1), the `explain()` plan and the `is_elementwise=False` prediction (beat 2), the
two-engine prediction (beat 3), the one-versus-ten prediction and the `0.0` on Polars 1.44 (beat 4), three-versus-eleven,
the `1 - (1 - p)` slide, the `-1.147` and "a report is a contribution" (beat 5), the disclaimers (beat 6).

## Sources

Every number and incident above, with where it comes from. "Re-run" means reproduced on the machine described in
"Benchmark conditions", on the date given.

| Claim | Source |
|---|---|
| `stats.norm(loc=0, scale=-1).sf(1.0)` is `nan`, no warning | Re-run 2026-10-03, scipy 1.18.1, warnings captured: none |
| R's `pnorm(q, mean, sd, lower.tail = FALSE)` is vectorised over every argument, so it works per row inside `mutate` | R documentation, `?Normal` (stats package). Not re-run here: R is not installed on the talk machine |
| Polars has no `erf` | Re-run 2026-10-03, Polars 1.44.2: no attribute containing `erf` on `pl.Expr` |
| `explain()` output, filter inside the Parquet scan; 482 rows identical on both engines | Re-run 2026-10-03, Polars 1.44.2, polars-stats `main`, 1,000,000-row file of the two tutorial sensors |
| Same query with the plugin registered `is_elementwise=False`: `FILTER col("sensor") == "temp-b"` stays above the plugin node instead of moving into the scan | Re-run 2026-10-04, Polars 1.44.2, `normal_sf` registered directly with `register_plugin_function` on a 1,000,000-row file |
| Branch masking needs the promise too: `when(x < 0).then(0).otherwise(normal_sf)` with `sigma = -1` on the `then` row returns a value with `is_elementwise=True` and raises `ComputeError` with `False`, on both engines | Re-run 2026-10-04, Polars 1.44.2. Supports beat 4's "each branch is evaluated only on the rows that select it"; not spoken |
| First commit: 26 April 2026, `Bernoulli.sample`, one `ChaCha20Rng` and a row loop, under seventy lines of Rust, the "not supported" docstring and the `pl.repeat` comment | PR #1 (`e0d4a16`); comment and docstring quoted verbatim |
| Single seeded stream made output depend on chunk boundaries; fixed by `(seed, index)` seeding before any release | PR #6 (2026-05-26); first PyPI release was 0.0.1 on 2026-08-09 |
| Per-row `ChaCha20` 10 to 20 times slower; `Pcg64Mcg` took 1M × 100 from about 5.9 s to 0.29 s | PR #7 (2026-06-01) |
| statrs samples Binomial in O(n); `rand_distr` instead | PR #28 (2026-06-11) |
| `samples` as one native call, `samples(size=1)` equals `sample` | PR #30 (2026-06-14) |
| CI runs every test on both engines | PR #57 (2026-08-21) |
| `simulated` column on the six-row frame, `ps.Normal(mu="mu", sigma="sigma").sample(seed=42)`: `9.763219`, `10.825292`, `11.172469`, `98.532788`, `100.147242`, `97.717269`, identical on both engines | Re-run 2026-10-04, polars-stats 0.1.0, Polars 1.44.2 |
| 0.0.1, `sigma=pl.lit(0.5)`, 1M rows: 1 distinct value in-memory, 10 streaming, height 1,000,000, no error; 0.0.2 correct | Re-run 2026-10-03 from PyPI packages in a clean environment outside the checkout, Polars 1.43.2 |
| Alignment moved to Rust, breaking release 0.0.2 | PR #58 (2026-08-24); 0.0.2 released 2026-08-30 |
| `over` example (per-sensor `mean` / `std` as parameters): `ShapeError` on 0.0.1 in-memory, correct on `main` on both engines | Re-run 2026-10-03: 0.0.1 from PyPI on Polars 1.43.2; `main` on Polars 1.44.2, agreeing with scipy within the parity suite's `1e-10` relative tolerance. Not claimed bit-identical across engines: Polars' own `mean` / `std` differ by a few ulps between engines |
| Polars 1.44 masks unselected `when / then` arms | Upstream Polars PR #28498; issue filed as Polars #29005 (open) |
| 0.0.1 (requires `polars>=1.15.0`, no cap): Uniform with `max < min` raises on Polars 1.43.2, returns `0.0` on 1.44.2 | Re-run 2026-10-03 from PyPI packages, plain `pip install` resolution |
| About ten days from cap to wrapper deletion | PR #75 (cap, 2026-08-29), #79 to #83 (port, lift), #88 (wrapper deleted, 2026-09-08) |
| Parity against scipy plus hypothesis invariants | PR #10 (2026-06-02) |
| mpmath at 50 digits; 11 defects against 3 budgeted; Bernoulli and Uniform 4, Exponential 3; nine one-to-three-line closed-form fixes; `Bernoulli(1e-300).sf(0.5)` was `0.0` | PR #43 (2026-08-09) and the maintainer's audit notes of 2026-08-07 (the count is not in a public PR body) |
| Sensor faults: `log_sf` `-1805` and `-2817`, true tails about `1e-784` and `1e-1224` | Docs tutorial, step 5 (executed at docs build) |
| polars_rng: no `seed` argument, thread-local generator | Its README and the comparison in the polars-stats docs (`docs/index.md`, Related projects); not installed here (no macOS arm64 package) |
| polars-random 0.5.0, `seed=42`, 1M rows: re-run same; in-memory vs streaming different (scalar and column parameters); 1 vs 4 chunks different with column parameters; polars-stats same in all three | Re-run 2026-10-03 in a clean environment, Polars 1.44.2 |
| `Beta(s, s).cdf(0.5)`: `0.4912`, `0.2129`, `-1.147` at `s = 1e6, 1e7, 1e8` | Re-run 2026-10-03 on `main` (statrs 0.19.1); also in the accuracy page |
| statrs reports (filed 2026-08-07 to 2026-09-18): fixed and released in 0.19.1 (2026-08-11): #432, #433, #437 via statrs PR #438 (the speaker's PR #436 was merged into it); fixed and merged after 0.19.1, unreleased: #434 (PR #456), #452 (PR #454); open: #435, #449, #471, #472, #473 | GitHub, checked 2026-10-03. Re-check the week before the talk: counts will move |
| Polars ships the plugin machinery | pyo3-polars lives in the `pola-rs` GitHub organisation; the plugin guide is part of the Polars user guide (see Links) |
| Contributors | polars-stats: sidsri14 (PRs #64, #68), camriddell (PR #110). statrs fixes for these reports: day01 (PR #438, #456), teddytennant (PR #454), merged by the maintainer YeungOnion; names for the credits slide if you want them there |
