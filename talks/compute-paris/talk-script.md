# Fifty-five lines of caveats

*Every bug `polars-stats` shipped was already in its first commit, written down as handled or "not supported".*
(Accepted as: "Vectorized statistical distributions, built on the Polars engine".)

**Thesis (one sentence):** When you build on two giants, a limitation you write down but cannot enforce from inside
the engine is not handled, it is a bug with a delivery date, and the history of this library is the list of those
dates.

**They believe:** two reasonable things. Statistics on Polars data means a round trip through `scipy.stats`, and that
is fine. And if you know a limitation and document it, or guard it in Python, you have handled it.

**It breaks when:** the first commit, 55 lines of Rust and 51 of Python, already says in a comment that a length-one
input would make the plugin draw once and repeat it, and says in a docstring that chunked input is "not supported".
The first became ten distinct answers for a million rows in a published release. The second meant the same seed gave
different numbers on the two engines. The third, two validators and no contract, was switched off from the outside by
a Polars optimisation.

**Do this instead:** borrow the maths (`statrs`), borrow the engine (Polars), keep every result a lazy expression that
runs on both engines, and move every contract to the one place that can enforce it: the first line of the Rust
function. Where you cannot enforce it, make it a red test, an upstream issue, and a documented regime with a
magnitude. Never a comment.

**Running example:** the first commit's Python file, `Bernoulli(p).sample(seed)`, opened at the start and reopened at
the end; and one lazy pipeline over a Parquet file of fleet telemetry (`device`, `value`, per-device `mu`, `sigma`)
scored with `Normal(mu, sigma).sf(value)`, filtered, and sunk back to Parquet on the streaming engine.

**Audience:** compute! Paris. Python data engineers and scientists who use Polars or pandas daily, know `scipy.stats`
by name, and have never opened a Rust file. · **Assumed knowledge:** Polars expressions, lazy frames, what a cdf is.

**Promise:** After this talk you will be able to look at your own "known limitation" comments and say which of them
are bugs with a delivery date, and what it costs to move each one inside the engine.

**Format:** 25 min talk + 5 min Q&A · live demo: no (recorded outputs on slides) · voice: calm contrarian ·
spine: received wisdom rethought, three threads pulled from one file, dates as evidence rather than structure

**Speaker's stance, stated once and early:** I am not a Rust expert, and a good part of the Rust layer was written
with AI assistance. That sentence is in the README. What I vouch for is behaviour, pinned by tests, and this talk is
about how each of those tests came to exist.

**Words this talk does not use:** "seam". The code between the two giants is "the boundary" or "the glue", and what
it enforces is "the contracts". Never a count of distributions, shipped or remaining.

## Outline

| # | Beat | Min | Setup (what they think now) | Turn (what breaks or flips) | Evidence |
|---|------|-----|------------------------------|------------------------------|----------|
| 1 | 26 April, 23:51 | 2.5 | Fifty-five lines of Rust, one method, tests pass. Looks done. Spot the bug. | Three spots are the whole story, and two are written down as handled or "not supported". | commit `0e66820`, 2026-04-26: `src/expressions.rs` (55 lines), `_bernoulli.py` (51 lines) |
| 2 | Why leave scipy, and what "lazy and streaming" actually buys | 4 | The round trip is what everyone does, scipy broadcasts, per-row parameters already work. | The cost is where the result lands: the plan ends, alignment is yours, `scale=-1` is `nan`. Keep it a `pl.Expr` and the scoring is one node in the plan: a filter moves below it into the scan, the same plan runs on both engines and sinks to Parquet. Two giants hand you engine and maths; the code you write is the boundary, and all three caveats live there. | `explain()` on polars 1.44.2; both engines equal; `sink_parquet`; scipy 1.18.1 `nan` |
| 3 | Caveat 1: "chunked / streaming inputs are not supported" | 3.5 | One `ChaCha20` stream, advanced per row. Documented limitation. | The caller cannot switch chunking off, and the streaming engine cuts differently: same seed, two engines, different numbers. Per-row `ChaCha20` fixed it at 10 to 20x; `Pcg64Mcg` on `(seed, row_index)` made it free. The index is the one input that must never broadcast, because the streaming engine hands the plugin one-row morsels. CI runs every test on both engines. | #5 (2026-05-26), #7 (2026-06-01, "10-20x regression"), #57 (2026-08-21); today: seeded column equal across engines |
| 4 | Caveat 2: "a length-1 `pl.lit(p)` would make the plugin draw once" | 3.5 | The comment names the mechanism and `pl.repeat` handles it. 0.0.1 ships. | A million rows scored, ten distinct answers, twelve with four threads. `pl.lit`, `.first()`, `.max()` are length one at runtime; Python cannot see it at plan time. A small test frame on the streaming engine came back at the right height and hid it. Alignment moved to Rust, line one. Breaking release. | 0.0.1 re-run today: `n_unique` 10 / 12; #58 (2026-08-24); 0.0.2 (2026-08-30) |
| 5 | Caveat 3: two validators, no contract | 4 | Null the bad row and keep running is friendly. A validating plugin inside a `when` arm enforces the raise. | Nulling rebuilds scipy's `nan`. Polars 1.44 masks and skips arms: 23 (distribution, method) pairs stopped validating, both engines. Cap, port the closed forms to Rust, lift the cap. Then: the port did not close the leak, because a Python null wrapper sat above every hook, Rust or not. Deleted. | #75 (2026-08-29), #79 to #83 (2026-09-05/06), #85 to #88 (2026-09-07/08); sweep 384 pass on 1.43.2, 106 fail on 1.44.1 |
| 6 | What was not written down | 3 | The maths is `statrs`, the closed forms are algebra I can read. Coverage is 95%. | A 50-digit audit found 11 defects where 3 were budgeted; the algebra was the worst offender (`1 - (1 - p)`), plus a panic and a `NaN`. And an unsupported dtype aborted the interpreter with no test failure. | audit 2026-08-07; #43; #84 (2026-09-06) |
| 7 | What it costs, honestly | 3 | Rust plugin, so faster than scipy, and every fix made things better. | 2 to 6x within one regime; scipy 10x faster on a constant `mean()`. The port made a `pl.lit` parameter 2.5 to 3.6x slower: the folding that made it fast hid the validator. The audit's non-`OK` count rose as the library improved. `Beta.ppf` waits upstream; `statrs` fixed three of our reports in 17 days. | benchmark 2026-09-22; port measurements at 10M rows; audit 2026-09-14; statrs#435; statrs 0.19.1 |
| 8 | Where it goes, and the lesson | 2.5 | | What remains needs new special functions, not more algebra. Three loops with no end. No known production user. Callback to the file: every caveat is now Rust line one, a red test, or a documented regime. | catalogue in the reference docs; accuracy page contract |

## Script

### Beat 1: 26 April, 23:51 (2.5 min)

**Setup:** a Sunday night in April 2026. The first commit of `polars-stats`: one distribution, one method,
`Bernoulli(p).sample(seed)`. Fifty-five lines of Rust, fifty-one of Python. Tests pass.

**Turn:** three spots in those 106 lines are the whole story of the project, and two of them are written down, one as
a handled case and one as "not supported".

**On screen:** the day-one Python file, condensed. Click 1 highlights the `pl.repeat` comment. Click 2 highlights the
docstring sentence "chunked / streaming inputs are not supported". Click 3 highlights the eager `ValueError` on a
Python float, next to the note that a column is checked in Rust with a different error.

**Say:**
Sunday, 26 April, eleven fifty-one at night. I push the first commit of a library called `polars-stats`. It does one
thing: draw a Bernoulli sample per row of a Polars frame, with a probability that can be a column. Fifty-five lines
of Rust. Fifty-one of Python. The tests pass.

Look at the Python for a moment, because I am going to ask you to find the bugs. There are three.

[click] Here is the first. A comment. It says, correctly, that Polars does not broadcast inputs into a plugin, so a
length-one `pl.lit(p)` would make the plugin draw once and Polars would repeat that one draw across the frame. So the
code pads every Python float with `pl.repeat`. Handled.

[click] Second. The docstring. "Reproducibility under seed assumes a single-chunk input series; chunked / streaming
inputs are not supported." A known limitation, written down.

[click] Third. A Python float is checked in Python, eagerly, with a `ValueError`. A column is checked in Rust, per
row, with a different error. Two validators, two messages, and nobody has decided which one is the contract.

Everything that went wrong in this project is on this slide. The first one shipped as a bug in a published release,
with a comment above it that describes the exact mechanism. The second one meant the same seed gave different
numbers on the two engines. The third one turned into the longest argument in the repository and a Polars release
that switched my validation off from the outside.

One sentence about me, because it makes the rest credible: I am not a Rust expert, and a good part of the Rust in
this library was written with AI assistance. It says so in the README. What I vouch for is behaviour, pinned by tests,
and this talk is the story of how each of those tests came to exist. By the end you will be able to look at your own
"known limitation" comments and tell which of them are bugs with a delivery date.

**Evidence:** commit `0e66820`, 2026-04-26 23:51 +0200, "feat: Bernoulli.sample". `src/expressions.rs` is 55 lines
(`ChaCha20Rng::seed_from_u64(seed)` once, then `for i in 0..n`), `polars_stats/distributions/_bernoulli.py` is 51 lines
and carries both the `pl.repeat` comment and the "not supported" docstring verbatim.

**If the demo fails:** n/a, pasted source.

### Beat 2: Why leave scipy, and what "lazy and streaming" actually buys (4 min)

**Setup:** the round trip through `scipy.stats` deserves a defence. `scipy` broadcasts parameter arrays, so a
different distribution per row is already vectorised there.

**Turn:** the cost is where the result lands. The plan ends at `.to_numpy()`, alignment is yours, and `scale=-1` is
`nan`. Keep the result a `pl.Expr` and the scoring becomes one node in the plan: a filter written after it moves below
it, into the Parquet scan; the same plan gives the same frame on the in-memory and the streaming engine; it sinks
straight back to Parquet without the frame ever fitting in memory. Two giants hand you all of that: Polars the engine,
`statrs` the maths. The code you write yourself is the boundary between them, and the longest files in the repository
have no maths in them. All three caveats live there.

**On screen:** slide 1, the round-trip code and the question `norm(loc=0, scale=-1).sf(1.0)` returns what, click:
`nan`. Slide 2, the lazy pipeline: `scan_parquet`, `with_columns(anomaly=...)`, two filters, then the real `explain()`
output with the device filter highlighted inside the scan, below the plugin node; click: both engines, equal, and
`sink_parquet`. Slide 3, two columns: what Polars hands a plugin, what `statrs` hands you. Slide 4, the tree at 0.1.0:
one maths file per distribution at about 300 lines, two boundary files at about 1000 lines with no maths.

**Say:**
Why write this at all? Before the library I did what you do. Pull the columns to NumPy, call `scipy.stats`, wrap the
result back into a Series. And that is a good idea. `scipy` broadcasts. If `mu` and `sigma` are arrays, `norm(loc=mu,
scale=sigma).sf(x)` scores every element against its own distribution, in C, no loop. Per-row parameters were solved
twenty years ago.

What is wrong with it is not speed. The plan ends there: you `collect()` before scoring, and the optimiser never sees
the second half. Alignment is yours: once the values leave the frame, keeping the result row-aligned through a join
or an `over` is your job. And NumPy has no null. Show of hands: `norm(loc=0, scale=-1).sf(1.0)` returns?

[click] `nan`. No warning. A negative standard deviation is a modelling error, and it travels through your pipeline
dressed as missing data. Hold that thought.

So the idea was to keep the result a `pl.Expr`, and I want to show you exactly what that buys, because "lazy and
streaming" is a slogan until you look at a plan. Here is a Parquet file of telemetry. Scan it. Add an anomaly score,
the upper-tail probability under each row's own Normal. Filter to one device. Filter to scores below one in a
thousand. Ask Polars to explain.

The scoring is one node. `normal_sf`, a function in a shared library, with three column inputs. Look below it. The
device filter I wrote after the scoring has moved into the scan, as a selection on the Parquet reader, so only that
device's row groups are decoded. The score filter stays above, because it depends on the score. Nobody materialised
anything.

[click] Now collect it twice, once on the in-memory engine, once on the streaming engine. Same frame, row for row.
Replace `collect()` with `sink_parquet()` and the whole thing streams from one file to another, in morsels, and the
frame never has to fit in memory. Every method in the library returns an expression, so this works for a sample, a
quantile, a log-density, anything. That is the whole product, and I did not write any of it.

[click] Here is what I did write, and what I did not. A Polars plugin is a Rust function with one attribute. It
receives Series, it returns a Series. For signing that contract you get lazy evaluation, the optimiser, the streaming
engine, the thread pool, and `over` and `group_by`, because Polars calls your function once per partition. That is
the first giant. The second is `statrs`, the Rust crate that does what `scipy.stats` does: `Normal::new(mu, sigma)`,
then `cdf`, `sf`, `inverse_cdf`, `ln_pdf`. Someone else got the error function right.

Look at the tree today. One Rust file per distribution, about three hundred lines, and the maths in it is a dozen
one-line calls into `statrs`. The two longest files in the repository, close to a thousand lines between them,
contain no maths at all. One aligns inputs and checks dtypes. The other decides how a row gets its seed. That is the
boundary between the two giants, it is the only code that is mine, and all three caveats from the first commit live in
it. Let us pull the first thread.

**Evidence:** `scipy` 1.18.1, `stats.norm(loc=0.0, scale=-1.0).sf(1.0)` is `nan`, warnings enabled, none emitted.
Pipeline run 2026-09-23 on polars 1.44.2, polars-stats 0.1.0, 1,000,000-row Parquet file with 40 devices:
`scan_parquet(...).with_columns(anomaly=ps.Normal("mu","sigma").sf("value")).filter(pl.col("device")=="sat-07")
.filter(pl.col("anomaly")<1e-3)`. `explain()`:

```text
FILTER col("anomaly") < 0.001
FROM
   WITH_COLUMNS:
   [col("value")..../_internal.abi3.so:normal_sf([col("mu"), col("sigma")]).alias("anomaly")]
    Parquet SCAN [readings.parquet]
    PROJECT */4 COLUMNS
    SELECTION: col("device") == "sat-07"
    ESTIMATED ROWS: 1000000
```

`collect(engine="in-memory")` and `collect(engine="streaming")` both return 911 rows and `.equals()` is `True`;
`sink_parquet` writes the same 911 rows. `Normal("mu","sigma").sample(seed=42)` collected on both engines is equal.
At v0.1.0: `src/distributions/normal.rs` 304 lines, `src/distributions/mod.rs` 464, `src/rng.rs` 501.

**If the demo fails:** pasted plan and outputs.

### Beat 3: Caveat 1: "chunked / streaming inputs are not supported" (3.5 min)

**Setup:** the first sampler is what anyone would write. One `ChaCha20` generator, seeded from your seed, advanced
once per row in iteration order. Reproducible, as long as the rows arrive in one chunk in one order. The docstring
says exactly that.

**Turn:** "not supported" is not a mode the caller can switch off. Polars chunks when it wants, the streaming engine
chunks differently from the in-memory one, and the thread pool decides the order. Same seed, same frame, two engines,
different numbers. The fix gave every row its own generator keyed on `(seed, row_index)`, and constructing a
`ChaCha20` per row made sampling ten to twenty times slower. `Pcg64Mcg` made the fix free. Two engine-specific facts
came out of it: the row index is the one input that must never be broadcast, because the streaming engine hands the
plugin one-row morsels and the flattened index is all it ever sees; and every test now runs twice, once per engine.

**On screen:** slide 1, prediction: same seed, same frame, in-memory and streaming, same numbers? Magic move from one
stream to the per-chunk reality. Slide 2, the fix and its bill: `row_rng(seed, index)`, "10-20x regression" from the
commit body, `Pcg64Mcg`, the index-must-not-broadcast wrinkle, and CI on both engines. Today's invariance line.

**Say:**
The docstring said chunked input was not supported. What can that sentence mean to a caller? You write
`sample(seed=42)`. Polars decides how many chunks your frame is in. The streaming engine cuts it into morsels, and it
cuts differently from the in-memory engine. The thread pool decides which chunk the plugin sees first. There is no
argument you can pass to make the input single-chunk. So "not supported" meant: the seed reproduces the chunk layout,
not the query. Prediction. Same seed, same frame, `collect()` twice, once per engine. Same numbers? Hands up.

No. Row zero of chunk three got the fourth draw on one engine and the four-thousandth on the other. And to be fair to
that first design, it is what anyone writes: one generator, advanced per row. It is correct for a list. It is wrong
for an engine that owns the iteration order, and the streaming engine owns it more than most.

The fix, a month in, is to stop having one stream. Every row derives its own generator from the seed and its global
position. Where does the position come from? Polars does not tell a plugin which rows it is holding. So the Python
side adds one input column, an integer range, and passes it in like any other parameter. The plugin never asks where
it is. It reads it.

Then the bill. A `ChaCha20` generator per row means a key schedule per draw, and the commit that fixed it a week later
says "10-20x regression" in its first line. Rust has cheaper generators. `Pcg64Mcg` is a handful of integer operations
to construct, it passes the usual statistical batteries, and its output is stable across releases and platforms. With
a mixing step on `(seed, index)`, the per-row version costs what the one-stream version did. Seeded values changed;
the contract did not.

Two things about the streaming engine came out of this thread, and both are now tests. First, the row index is the
one input that must never be broadcast. If it arrives as length one, every row is row zero, and you get one draw
repeated at full height with no error. The plugin cannot repair that, because the streaming engine splits the call
into one-row morsels and a flattened index is all it ever sees. So the index is sized by the call, in Python, before
anything crosses the boundary. Second, since August the test suite runs twice in CI, once per engine, because they
chunk a plugin's inputs differently and a bug at a chunk boundary passes on one and fails on the other.

Today: one chunk or ten, in-memory or streaming, macOS or Linux, same column. `samples(size=1)` equals `sample` bit
for bit, because it is the same stream. That docstring sentence is gone. It became a property test.

**Evidence:** #5 (2026-05-26) "Make Bernoulli sampler truly elementwise via per-row sub-seeds"; #7 (2026-06-01)
"Replace per-row ChaCha20 construction (10-20x regression) with a shared Pcg64Mcg helper ... seeded output values
change; behavioural contract unchanged." #57 (2026-08-21) "Run tests for both engines". The one-row-morsel fact and
the `row_index_expr(params)` sizing are in the architecture page, "Sampling". Verified 2026-09-23 on v0.1.0:
`Normal("mu","sigma").sample(seed=42)` from `scan_parquet` collected on in-memory equals streaming;
`samples(1, seed=42).arr.first()` equals `sample(seed=42)`.

**If the demo fails:** pasted outputs.

### Beat 4: Caveat 2: "a length-1 `pl.lit(p)` would make the plugin draw once" (3.5 min)

**Setup:** the comment names the mechanism and `pl.repeat` handles it. Version 0.0.1 ships in August, mostly to claim
the name. A million telemetry rows, each with its own baseline, scored in one line.

**Turn:** the result column has ten distinct values. Twelve with four threads. No error. `pl.repeat` padded Python
floats. `pl.lit(0.5)`, `pl.col("sigma").first()` and `.max()` are length one at runtime, and Python cannot know that
at plan time. The zip truncates to one row and `with_columns` broadcasts it once per chunk. A small test frame on the
streaming engine had come back at the right height, one output row per one-row morsel, and hid it. Alignment moved
into Rust, on the first line, by length rather than by expression kind. Breaking release.

**On screen:** slide 1, the expression, click reveals `n_unique() = 10`, click `12`. Slide 2, the day-one comment
again beside the four spellings and which it covered, plus the streaming-hid-it line. Slide 3, magic move from the
Python padding to the Rust `align_inputs`, and the breaking change.

**Say:**
Second thread, the comment. Version 0.0.1 goes to PyPI in August. I shipped it early, mostly to claim the name.

Here is the telemetry frame again, a million rows, every row with its own baseline. One line scores it. A million
rows in, a million out, `Float64`, right length.

Count the distinct values in that column. [click] Ten. Run it with four threads instead of ten. [click] Twelve.

Go back to the comment. "A length-one `pl.lit(p)` would make the plugin draw once and Polars would repeat it across
the frame." I wrote that on day one. The code padded every Python float with `pl.repeat`, and a Python float was fine.
What I had not thought through is that a float is not the only way to hand me a length-one input. `pl.lit(0.5)` is
length one. `pl.col("sigma").first()` is length one. `.max()` is length one. And Python cannot know that when it
builds the plan, because the length of `.first()` is a runtime fact.

So the plugin received a million values, a million means and one sigma. The helper that walks them together is a zip,
and a zip stops at the shortest input. It returned one row, correctly computed, and `with_columns` did what it does
with a one-row result: broadcast it. Once per chunk. Ten chunks, ten answers.

And here is the part I find most uncomfortable. I had tested this. On a small frame. On the streaming engine, which
hands the plugin one row at a time, so one output row per one-row morsel adds up to exactly the frame height. The
test passed. A small frame on the streaming engine is not evidence of anything about lengths.

The fix is small and it is in Rust, because Rust is where the lengths are known. `align_inputs` runs before any cast.
Every length-one input is broadcast up to the row count of the call; two inputs of different lengths that are not one
raise a shape error instead of zipping. It aligns by length, not by what the expression looked like in Python.

It was a breaking release. If every input is constant, `Normal(0.0, 1.0).mean()` in a `select` is now one row, which
is what Polars does with a constant. The padding had made it a full column. The new behaviour is right and it was
still a change.

I had handled the caveat where I could see it, in Python, at plan time. It was only true in Rust, at runtime.

**Evidence:** #58 (2026-08-24), shipped in 0.0.2 (2026-08-30). Re-run of `polars-stats==0.0.1` with `polars==1.43.2`:
`Normal(mu="mu", sigma=pl.lit(0.5)).sf("value")` inside `with_columns` on 1M rows gives `n_unique` 10 with 1 or 10
threads and 12 with 4 threads; a Python scalar `sigma=0.5` gives 1,000,000. Streaming hid it at small sizes: measured
2026-08-21 with a plain `map_batches` zip, 10k rows gives height 1 on in-memory and the frame height on streaming,
because one row per one-row morsel adds up. On 0.1.0 `pl.col("sigma").first()` gives 1,000,000 distinct values.

**If the demo fails:** pasted outputs.

### Beat 5: Caveat 3: two validators, no contract (4 min)

**Setup:** the friendly answer, null the bad row and keep running, rebuilds the `scipy` `nan` problem inside Polars.
So the contract flipped: null in, null out; invalid parameter, raise, whole evaluation, scalar and column alike, from
one check. A `pl.Expr` cannot raise per row, so the closed-form distributions routed their parameters through one
small validating plugin, placed inside the `when/then/otherwise` that assembled the formula.

**Turn:** Polars 1.44 optimised `when`. An arm is masked on the rows it does not select and skipped if no row selects
it. The validator was in an arm. Twenty-three (distribution, method) pairs stopped validating, on both engines. Cap
`polars<1.44`, port the four Python-assembled distributions into Rust, lift the cap. Then the second turn: the port
did not close the leak. A Python wrapper, `when(value.is_null()) ... otherwise(hook)`, sat above every hook, including
the ones that had always been Rust, and on a null evaluation point the plugin never saw the parameters. Deleted. The
plugin owns every row.

**On screen:** slide 1, the two contracts side by side and the flip. Slide 2, `Uniform.cdf` assembled in Python with
the validator in the `otherwise` arm; click: 1.44 masks the arm; the sweep numbers; the two moves. Slide 3, the
wrapper above the hook with "Rust or not" underlined, the deletion, today's output.

**Say:**
Third thread: two validators, one in Python for floats and one in Rust for columns, and no decision about which one
was the contract. The decision is about what should happen when `sigma` is negative on row four hundred.

My first answer was the friendly one. Null the row. Keep the nightly job running. Then I looked at the result and
realised I had rebuilt the `scipy` problem from earlier: a row that is null because `sigma` was missing and a row that
is null because `sigma` was negative are the same row. One is missing data. The other is a bug upstream. So the
contract flipped. Null in, null out, always. Invalid parameter, raise, fail the whole evaluation, name the parameter
and the value. Same rule for a float and for a column, because both go through one check, in Rust.

Then the mechanism, because a decision you cannot enforce is a comment. A `pl.Expr` cannot raise per row. So for the
distributions whose closed forms were Polars arithmetic, I routed the parameters through a tiny Rust plugin whose only
job was to raise, and I put it where it seemed to belong: inside the `when / then / otherwise` that assembled the
formula.

[click] Polars 1.44 shipped an optimisation, and a good one. A `when` arm no longer sees rows it does not select, and
an arm nobody selects is not evaluated at all. My validator was in an arm. On a row where the condition sent the value
elsewhere, it received nothing, said nothing, and the method returned a number for a negative parameter. Twenty-three
distribution and method pairs, on both engines. Exactly the distributions whose closed forms lived in Python, and only
those. The ones that had always been Rust lost nothing. The sweep that found it passes 384 assertions on 1.43 and
fails 106 on 1.44.

Two moves. Cap Polars below 1.44 and ship, with a test that goes red the day the cap is lifted. Then move every
value-keyed closed form into Rust, where the check runs on the parameter column before any row is built, and lift the
cap. Two weeks. Done.

[click] Except it was not. Above every public method, ported or not, sat one more Python expression: when the value
is null, then null; when it is NaN, then NaN; otherwise, the plugin. Friendly, uniform, and an arm. On a null
evaluation point the plugin never ran, so a negative `sigma` on that row was never seen, for every distribution,
including the ones that had been Rust from the start. Moving the maths into Rust changed nothing about a wrapper that
sat above the maths. The wrapper went. The plugin answers the null and `NaN` rows itself, before it computes anything.

That is the rule from this thread, and it is the docstring from April coming back: a contract you cannot enforce from
inside the engine is a comment. It may hold for a while. It has a delivery date.

**Evidence:** design notes "Invalid parameters raise, they never silently null"; #65 (2026-08-25). pola-rs/polars#28498
merged 2026-08-11, in 1.44.0; polars#29005 open as of 2026-09-22. 23 affected pairs: Bernoulli 6, Geometric 6,
Uniform 6, Exponential 5; bisected clean on 1.43.2, reproducing on 1.44.1, both engines. #75 caps `polars<1.44.0`
(2026-08-29). #79 lifts it, #80 to #83 port the closed forms (2026-09-05/06). #85 to #88 (2026-09-07/08) move the
null/`NaN` contract into Rust and delete the wrapper. Current output: null `sigma` row is null; `sigma=-1` raises
`ComputeError: sigma must be finite and strictly positive, got -1`.

**If the demo fails:** pasted outputs.

### Beat 6: What was not written down (3 min)

**Setup:** three caveats were in the file. Two things were not, because I did not know to write them. The maths is
`statrs`, so the hard part is someone else's, and the closed forms are algebra I can read on a page. Coverage is 95%.

**Turn:** an audit against an `mpmath` oracle at fifty digits found 11 defects where 3 were budgeted. The special
functions I had been nervous about were fine. The elementary algebra I had trusted was the worst offender by count,
`1 - (1 - p)` for a small `p`. Two findings were worse than a wrong digit: `Beta.ppf` panicked, which aborts the whole
query, and `Binomial.entropy` returned `NaN`. And a `Decimal` or `Categorical` value column did not raise, it aborted
the interpreter, and a dead process reports no test failure, so coverage is not evidence at an FFI boundary.

**On screen:** slide 1, budgeted 3, found 11; click: where they were; click: the two that were not inaccuracy. Slide
2, the interpreter abort: no traceback, 95% coverage, the Cargo feature and its 3.4 MiB.

**Say:**
Those three were written down. Two more things were not, because I did not know to write them.

My confidence was in the wrong place. The maths I was nervous about was the special-function maths: the error
function, the incomplete beta. That was `statrs`, and I had read enough of it to trust it. The maths I was relaxed
about was the algebra. `Uniform.cdf` is `(x - a) / (b - a)`. `Geometric.sf` is `(1 - p) ** k`. You can check that on
paper.

In August I stopped checking on paper and swept every method against `mpmath` at fifty digits, at inputs many decades
past where `scipy` itself saturates. I had budgeted for three defects. It found eleven.

[click] The special functions came back fine. The elementary closed forms were the worst offenders by count. Nobody
had thought to check `1 - (1 - p)` for a `p` of ten to the minus twelve. It is exactly zero in `float64`. The tail was
right on paper and gone in the machine.

[click] And two of the eleven were not inaccuracy at all. `Beta.ppf` far in the lower tail panicked, and a panic
inside a plugin does not fail a row, it aborts the query. `Binomial.entropy` returned `NaN` on valid parameters. A
curated grid of test points had never reached either. An argument that a method is safe is a hypothesis, and the
sweep is the experiment.

The second thing nobody wrote down. In September a `Decimal` or `Categorical` value column did not raise. It aborted
the interpreter. No traceback, no test failure, because a dead process reports nothing. Coverage was at 95 percent,
and it was not evidence of anything at that boundary. The fix is a Cargo feature that makes an unsupported dtype
raise, and it costs three and a half megabytes of binary. Cheap, for the difference between an exception and a
disappearing process.

**Evidence:** audit run 2026-08-07: 11 fixable defects against a budget of 3, five of them scipy-parity failures the
existing grids had not reached; `Beta.ppf` panic and `Binomial.entropy` `NaN` categorically worse; nine of the eleven
were one- to three-line closed-form fixes. Shipped as #43 (2026-08-09). #84 (2026-09-06) `dtype-full`, +3.44 MiB.

**If the demo fails:** n/a.

### Beat 7: What it costs, honestly (3 min)

**Setup:** it is a Rust plugin, so it is faster than `scipy`, and every fix in this story made the library strictly
better.

**Turn:** within one regime, on one laptop, against the frozen `scipy` API, 2 to 6x less wall time; on a
constant-parameter `mean()` `scipy` is 10x faster. The Rust port made a `pl.lit` parameter 2.5 to 3.6x slower than a
Python float, because the constant folding that made it fast was the same folding that hid the validator. The audit's
count of non-`OK` probes went up as the library improved, because the instrument improved faster than the code.
`Beta.ppf` in the far lower tail does not return, filed upstream, not patched; and `statrs` fixed three of our reports
within seventeen days.

**On screen:** slide 1, five benchmark rows with the disclaimers printed larger than the numbers. Slide 2, the bills:
the `pl.lit` regime after the port, the audit count with its date, statrs#435 and the three fixed reports.

**Say:**
The numbers, disclaimers first because they matter more.

One laptop, an Apple M5, run this week. Polars 1.44.2, scipy 1.18.1, release build. One million rows. Wall time only,
median of fifty, no memory figures. The `scipy` side is the classic frozen API, which is what most code looks like and
is not `scipy` at its fastest; the newer random-variable API measured earlier at roughly 1.3 to 1.9x quicker on these
methods. The Polars side includes planning and `collect()`. Read a cell only against the cell beside it.

Within those limits: `sf` on a million rows with column parameters, about 3 milliseconds against about 13. Sampling,
about 2.5x. And the cell I like most: a constant-parameter `mean()`. `scipy` wins by 10x, because it returns the
parameter you gave it and a Polars query costs a tenth of a millisecond to plan. If you call a moment on constants a
million times in a loop, `scipy` is the right tool and this library is the wrong one.

Now the bills nobody puts on the benchmark slide.

The port that fixed the validator made one regime slower. A `pl.lit` parameter had been fast because Polars
constant-folded it through pure arithmetic, and that folding is exactly what stopped the validator from running. After
the port, `pl.lit` takes the per-row path and costs two and a half to three and a half times a Python float. The speed
and the bug had one root. Buying the speed back meant keeping the bug. I kept the fix and wrote the number down.

The audit. In July the headline was "34,000 probes, zero non-OK". In mid-September it was about 1,057 non-OK rows out
of 34,757. The library got better in between. The instrument got better faster: more distributions, wider sweeps,
inputs decades further into the tails. Every one of those rows maps to a documented caveat with a regime and a
magnitude, and that mapping is the bar, not zero. Do not quote the July number. I am telling you so you do not.

Last bill: what you inherit from a giant. `Beta.ppf` below ten to the minus one-sixty-five panics; between ten to the
minus one-fifty and ten to the minus sixty at some shapes it does not return, fifteen seconds for a single row. That is
filed upstream as statrs 435, it is in the docs with the regime and the magnitude, and it is not patched here. The
deal runs both ways: `statrs` 0.19.1 fixed three defects we had reported, seventeen days after 0.0.1. When 435 lands,
the bullet comes out.

**Evidence:** benchmark run 2026-09-22 on macOS 26.6.2, Apple M5 (10 cores, 24 GiB), CPython 3.12.13, polars 1.44.2,
scipy 1.18.1, numpy 2.5.3, polars-stats 0.1.0 release; `tools/benchmarks/run.py normal --regimes scalar column --rows
1_000_000`, p50 of 50. Table:

| regime | method | polars-stats p50 (ms) | scipy frozen p50 (ms) | ratio |
|---|---|---|---|---|
| column | sample | 1.66 | 4.26 | 2.6 |
| column | samples (10) | 8.23 | 37.40 | 4.5 |
| column | pdf | 1.24 | 5.87 | 4.7 |
| column | log_pdf | 1.58 | 8.63 | 5.5 |
| column | cdf | 2.93 | 12.97 | 4.4 |
| column | log_cdf | 3.66 | 19.85 | 5.4 |
| column | sf | 2.90 | 13.17 | 4.5 |
| column | log_sf | 3.67 | 19.61 | 5.3 |
| column | ppf | 2.73 | 14.11 | 5.2 |
| column | isf | 2.72 | 14.20 | 5.2 |
| column | mean | 0.84 | 2.59 | 3.1 |
| column | entropy | 1.52 | 6.74 | 4.4 |
| scalar | sample | 1.68 | 3.55 | 2.1 |
| scalar | sf | 2.16 | 12.23 | 5.7 |
| scalar | log_sf | 3.49 | 18.76 | 5.4 |
| scalar | mean | 0.11 | 0.01 | 0.08 |
| scalar | variance | 0.11 | 0.01 | 0.08 |

Port cost, measured 2026-09-05/06 at 10M rows: `column` regime -38% to -81% (Bernoulli) and -73% (Exponential);
`broadcast` regime (a `pl.lit` parameter) runs at 2.5 to 3.6x the `scalar` regime on the already-ported control
(`DiscreteUniform.pmf` 2.23 / 6.99 / 8.07 ms scalar / column / broadcast). Audit 2026-09-14: ~1057 non-`OK` of 34,757
probes, `--skip Beta.ppf Beta.isf`; `<TODO: re-run make audit on the presented release and refresh both numbers>`.
statrs#435 open as of 2026-09-22; statrs 0.19.1 (2026-08-26, #67) fixed `Beta` endpoint densities, `Beta.sf`'s lower
corner, `Binomial.entropy` `NaN`.

**If the demo fails:** n/a.

### Beat 8: Where it goes, and the lesson (2.5 min)

**Setup:** version 0.1.0 is out. The catalogue is in the reference docs, and it will have grown by the time you read
this.

**Turn:** what remains is not more of the same. Everything shipped so far needed at most an error function or an
incomplete beta; what is left needs new special functions, an incomplete gamma and a log-space incomplete beta, and
the second one also fixes the last two bad tails in the shipped catalogue. Then three loops with no end: dependency
bumps, the audit, the performance protocol. And the ugliest line in the plan: I do not know of a single production
user, so every priority is inferred from the code. Callback: the 55 lines.

**On screen:** slide 1, where it goes: the shape of the remaining work (special functions, not algebra) and the three
loops, with the no-user line in a box. Slide 2, the day-one file again, each caveat replaced by where it lives now.
Slide 3, the lesson.

**Say:**
Where it stands. Version 0.1.0 is out. The catalogue is in the reference docs, and I am not going to read it to you,
because it will have changed by the time this recording is online.

Where it goes. What is left is not more of the same. Everything shipped so far needed, at most, an error function or
an incomplete beta, and `statrs` had both. What remains needs special functions the library does not bind yet: an
incomplete gamma for one family, and a log-space incomplete beta for another. That second one also fixes the last two
bad tails in what is already shipped, the `log_cdf` of `Beta` and `Binomial`. It is written. It is parked, because the
decision to report upstream rather than patch locally means waiting for `statrs` to say where it should live.

Then three loops with no end state. Every `statrs` or Polars release gets a bump and a re-audit, and every workaround
an upstream fix makes redundant gets deleted. Every new distribution gets the fifty-digit sweep before it merges, not
after. And every performance change gets an A/B against an untouched control on a quiet machine, because I have
already been fooled once by a block order that faked a twenty percent regression.

And the honest line at the bottom of the plan: I do not know of a single production user. Every priority I just gave
you is inferred from the code. The first real workload will reorder something, and I would like it to be yours.

Back to the twenty-sixth of April. Fifty-five lines of Rust, fifty-one of Python, three caveats. Here is where each
one lives today. The padding comment became `align_inputs`, the first line of every plugin, and a broadcast test that
sweeps `pl.lit`, `.first()` and `.max()` on both engines. The "not supported" docstring became a per-row seed and a
property test across chunk layouts and both engines. The two validators became one, in Rust, that owns every row, with
a sweep that went red the day Polars changed. Not one of them is a comment anymore.

So the rule, whatever your two giants are. A limitation you write down but cannot enforce from inside the engine is
not handled. It is a bug with a delivery date. Move it inside, or make it red.

Borrow the maths. Borrow the engine. Own the contracts. `pip install polars-stats`. The design notes have every one of
these mistakes written up with the date it shipped and the date it was fixed. Thank you.

**Evidence:** v0.1.0 tagged 2026-09-20; the catalogue is `polars_stats.__all__` and the reference docs, never a count.
Remaining work needs the regularized incomplete gamma (one family, and two others delegate to it) and a log-space
regularized incomplete beta (two families, plus `Beta` / `Binomial` `log_cdf` / `log_sf`; tracked as statrs PR #421,
written and parked on a branch, not on `main`). Standing loops as recorded in the project's planning notes (not in the
repository). "No known production user" is the maintainer's own statement as of 2026-09-15; `<TODO: speaker to
confirm the week of the talk, or soften to "I have not heard from one">`.

## Takeaways (max 3, one is better)

1. A limitation you write down but cannot enforce from inside the engine is a bug with a delivery date. Move it
   inside, or make it red.

## Links & resources

* `pip install polars-stats` · <https://github.com/FBruzzesi/polars-stats>
* Docs, design notes and the accuracy page: <https://fbruzzesi.github.io/polars-stats/>
* Polars plugins: <https://docs.pola.rs/user-guide/plugins/> · `pyo3-polars`: <https://github.com/pola-rs/pyo3-polars>
* `statrs`: <https://docs.rs/statrs> · statrs#435 (Beta inverse) · polars#29005 (`when` arm masking)

## Cut list

Every talk runs long. Cut in this order, setup before turn:

1. Beat 7, the audit-count paragraph and the `statrs` 0.19.1 aside. Keep the benchmark slide with its disclaimers and
   the `pl.lit` bill. Saves ~1 min.
2. Beat 8, the three standing loops. Keep the shape of the remaining work and the callback. Saves ~45 s.
3. Beat 2, the two-giants slide. The plan slide and the tree-shape slide carry the point. Saves ~45 s.
4. Beat 3, the `Pcg64Mcg` detail. Keep "10 to 20x slower" and "a cheaper generator made it free". Saves ~30 s.
5. Beat 6, the interpreter-abort paragraph. Saves ~40 s.

Never cut: beat 1's three clicks, the `explain()` slide in beat 2, beat 4's reveal, the second turn of beat 5 (the
port that did not fix it), the callback slide in beat 8.

## Pre-talk TODOs

* Re-run `tools/benchmarks/run.py normal --regimes scalar column --rows 1_000_000` on the release you present, on a
  quiet machine, and refresh the five rows on the benchmark slide and the environment line.
* Re-run `make audit` on the presented release and replace the 2026-09-14 figures (~1057 non-`OK` of 34,757) with
  the current pair, dated.
* Re-run the `explain()` pipeline on the presented polars version; the plan text may change shape.
* Re-check statrs#435 and polars#29005 the week of the talk; both were open on 2026-09-22.
* Confirm or soften "I do not know of a single production user" (beat 8).
* Re-check the line counts on the tree-shape slide against the release you present, or drop the numbers and keep the
  shape.
