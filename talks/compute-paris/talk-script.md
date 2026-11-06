# Vectorized statistical distributions, built on the Polars engine

The outline of the compute! Paris 2026 talk: what it argues, how it gets there, and where every number on a slide comes
from. The words themselves are the presenter notes in [`slides.md`](slides.md).

## The argument

**In one sentence.** Polars hands you the engine and statrs hands you the maths; every bug that mattered in
`polars-stats` lived in the thin layer between them, in four promises neither of them makes: reproducibility,
broadcast, parameter validation and accuracy.

**The lesson, as one instruction.** Borrow the engine, borrow the maths, and enforce what neither promises you inside
the plugin, before it becomes a bug.

**The three libraries.** Each is introduced, on its own slide, at the moment the story needs it:

| Library | What it is | What it gives `polars-stats` |
|---|---|---|
| Polars | A dataframe query engine written in Rust, driven mostly from Python | Lazy plans, the optimiser, both engines, `over` and `group_by`, the thread pool, and expression plugins |
| scipy | Python's scientific library; `scipy.stats` has more than a hundred distributions | The vocabulary (method names) and the ruler (every method is tested against it) |
| statrs | The closest thing Rust has to `scipy.stats`, maintained by volunteers | The maths: densities, cdfs, inverses, sampling, with `erf`, gamma and beta underneath |

**The running example.** Six readings from two temperature sensors, two of them faults. Two jobs: score each reading
against its own sensor's distribution (`sf`, then `log_sf`) and simulate readings from it (`sample`). The sensor table
gains a column as the talk goes, and a stack (Polars on top, statrs below, `polars-stats` in between) gains one
contract per section.

**Audience and format.** People who work with open-source computation across languages. Nothing about Python, Polars,
scipy, Rust or statrs is assumed; every term is defined the first time it is used. 25 minutes plus 5 of Q&A, no live
demo: every output on a slide is a recorded, re-run result.

## Question chain

Each section opens with its question on a divider slide, and each answer forces the next question.

| Slides | Section | Question |
|---|---|---|
| 1-2 | Opening | Each sensor has its own baseline: how do I score and simulate readings, row by row? |
| 3-7 | 01 · Two libraries | Polars has the data, scipy has the maths: why not use both? |
| 8-14 | 02 · The plugin | Can a distribution be a Polars expression instead? |
| 15-18 | 03 · Reproducibility | Same seed, same column? |
| 19-22 | 04 · Broadcast | Does every input arrive at full length? |
| 23-26 | 05 · Parameter validation | Does every row reach the parameter check? |
| 27-33 | 06 · Accuracy | Is the number on each row right? |
| 34-37 | 07 · Cost | Was it worth it, and what does it cost? |
| 38-40 | Close | The stack with all four contracts, the lesson, links |

Retelling test, what an attendee says at lunch: "They built a statistics library on Polars and a Rust maths crate,
with scipy as the blueprint and the ruler. The engine and the maths came for free; every bug was in the little bit of
glue in between, in things nobody had promised."

## Takeaways

1. **Borrow the engine, borrow the maths, and enforce what neither promises you inside the plugin.**
2. **A plugin sees pieces, not your query.** Key anything stateful, like a seed, on row identity, and align inputs by
   their length at run time.
3. **Your own algebra is maths too, and a bug report is a contribution.** Check both against an oracle with more digits
   than either library, fix what is yours, and send what is upstream's upstream, with a reproducer.

## Timing and cut list

At 130 words a minute the notes run about 26 minutes. Section marks: opening ends 1:25, 01 at 4:55, 02 at 10:10, 03 at
12:45, 04 at 15:00, 05 at 17:40, 06 at 22:05, 07 at 24:45, close at 26:00.

Cut whole asides, never a wall or a prediction, in this order:

1. The cost of a per-row generator on slide 18.
2. The R sentence on slide 11.
3. The `sink_parquet` / `over` / `group_by` sentence on slide 13.

Never cut: the `nan` quiz, the plan with and without `is_elementwise`, the two-engine prediction, the one-versus-ten
prediction, the `0.0` on Polars 1.44, the `1 - (1 - p)` derivation, `-1.147`, and the benchmark disclaimers.

## Sources

| Claim | Source |
|---|---|
| `stats.norm(loc=0, scale=-1).sf(1.0)` is `nan`, with no warning | Re-run with scipy 1.18.1, warnings captured: none |
| Polars has `erf` and `erfc` but no gamma or beta functions, inverse cdfs or distribution sampling | Polars 2.1.0, `dir(pl.Expr)`; `Expr.sample` samples rows, not distributions |
| statrs `Normal::new(10.0, 0.5).sf(10.4)` is 0.211855… and `Normal::new(0.0, -1.0)` is `Err(StandardDeviationInvalid)` | statrs 0.19.1: the value through polars-stats, which calls `sf` directly; the error from statrs' source |
| The pushdown plans, with and without `is_elementwise`, and 497 identical rows on both engines | Re-run on Polars 2.1.0, a 1,000,000-row Parquet file of the two sensors; the plugin node is shortened on the slide |
| The `simulated` column, `seed=42`: `9.763219`, `10.825292`, `11.172469`, `98.532788`, `100.147242`, `97.717269`, identical on both engines | Re-run on Polars 2.1.0 |
| The first sampler: one `ChaCha20Rng` walking the rows, documented as not supporting streaming | The first commit, `e0d4a16` |
| A generator per row: `ChaCha20` 10 to 20 times slower; `Pcg64Mcg` took 1M rows × 100 draws from 5.9 s to 0.29 s | PR #7 |
| `polars-random` 0.5.0 gives different seeded columns on the two engines and across chunk layouts; `polars_rng` takes no seed | Measured with `polars-random` 0.5.0 on Polars 1.44.2; `polars_rng` from its README |
| 0.0.1 with `sigma=pl.lit(0.5)` on 1M rows: 1 distinct value in memory, 10 streaming, no error | Re-run from PyPI in a clean environment, Polars 1.43.2 |
| Alignment by length moved to Rust, a breaking release | PR #58, released as 0.0.2 |
| 0.0.1's Uniform cdf, with `max < min`: `ComputeError` on Polars 1.43.2, `0.0` on 1.44.2 | Re-run from PyPI; the code is `polars_stats/distributions/_uniform.py` at `v0.0.1` |
| Polars 1.44 evaluates a `when` branch only on the rows it selects | Polars PR #28498; reported as pola-rs/polars#29005 |
| The invalid-parameter message: `sigma must be finite and strictly positive, got -1` | Re-run on `main` |
| `Bernoulli(p=1e-300).sf(0.5)` returned `0.0` and now returns `1e-300` | Fixed in PR #43, found by the mpmath audit (`make audit`); re-run on `main` |
| The two faults: `log_sf` `-1805.01` and `-2817.74`, true tails about `1e-784` and `1e-1224` | Docs tutorial, step 5; re-run on `main` |
| `Beta(s, s).cdf(0.5)` in statrs 0.19.1: `0.4912`, `0.2129` and `-1.147` at `s = 1e6, 1e7, 1e8` | statrs issue #434, fixed by statrs PR #456; also in the accuracy page |
| The benchmark numbers and their conditions | `tools.benchmarks.run`, column regime, run on 3 November 2026; the command is in the README |
| scipy's newer API is 1.3 to 1.9 times faster than the frozen one on the Normal | Measured earlier with the same harness, column parameters |

## Links

* Docs: <https://fbruzzesi.github.io/polars-stats/>, start with Explanation
* Code: <https://github.com/FBruzzesi/polars-stats>, `pip install polars-stats`
* scipy.stats: <https://docs.scipy.org/doc/scipy/reference/stats.html>
* Polars expression plugins: <https://docs.pola.rs/user-guide/plugins/>
* statrs: <https://docs.rs/statrs>
* pyo3-polars: <https://github.com/pola-rs/pyo3-polars>
* mpmath: <https://mpmath.org>
