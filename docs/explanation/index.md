---
icon: lucide/compass
---

# Why polars-stats

Understanding-oriented pages: what problem this library solves, how it is put together, and why. Nothing here is
needed to *use* the library.

## The problem

Statistical work on Polars data usually means leaving Polars. You call `.to_numpy()`, hand the arrays to
`scipy.stats`, and build a `pl.Series` from the result. The alternatives are worse: a Python UDF through
`map_elements` is slow and holds the GIL, and hand-rolled per-distribution expressions are ad hoc and easy to get
subtly wrong in the tails.

## What scipy already does well

The gap is smaller than it first appears. `scipy` broadcasts parameter arrays perfectly well:
`stats.norm(loc=mu_array, scale=sigma_array).sf(x_array)` scores every element against its own distribution,
vectorised, with no Python loop. Per-row parameters are not the missing piece.

What is missing is that the result is a NumPy array rather than a `pl.Expr`. That difference costs you four things:

* **The round trip.** Every parameter column and every evaluation point is materialised to NumPy and the result is
    wrapped back into a `Series`. On a `LazyFrame` you must `collect()` first, so the query is cut in half.
* **The optimiser.** A `collect()` in the middle means predicate pushdown, projection pushdown, and streaming stop at
    that boundary. Filtering after scoring cannot be pushed before it.
* **Alignment.** Once the values leave the frame, keeping the result row-aligned through joins, filters, and
    partitions is the caller's problem. Inside `over` or `group_by` it becomes genuinely awkward.
* **Missing data.** NumPy has no null. `scipy` conflates "missing" with `NaN`, and an invalid parameter such as
    `scale=-1` silently returns `NaN` instead of raising, so a modelling error can travel a long way before anyone
    notices.

## What this library does about it

`polars-stats` makes a distribution an expression builder. Every method returns a `pl.Expr`, so it composes in a lazy
plan; every parameter accepts a column, so one instance describes a different distribution per row; nulls stay nulls;
and an invalid parameter raises rather than nulling. The math itself is delegated to the
[`statrs`](https://docs.rs/statrs) crate rather than reimplemented.

## What it costs

* **A much smaller catalogue.** A short list against scipy's hundred-plus, no multivariate distributions, and no
    hypothesis tests. The [catalogue](../reference/index.md#catalogue) is what ships today, and it is still growing.
* **No estimation API.** There is no `fit`; you write the estimator as a Polars expression.
* **No `loc` / `scale` shims.** Shifting or scaling an arbitrary distribution is the caller's job.
* **An FFI boundary per method call.** Even a closed-form `mean()` makes one round trip into Rust, to keep parameter
    validation uniform.
* **A compiled dependency.** A Rust extension module, and exposure to `pyo3-polars` ABI churn across Polars releases.

Where a distribution or a method is missing, `scipy` remains the right tool, and both can live in one project. See
[Migrate from scipy.stats](../how-to/migrate-from-scipy.md) for the mapping.

## Read on

* [Architecture](architecture.md): what the system is and how it is wired, from the Python distribution classes down
    to the Rust plugins: the layer split, plugin granularity, the constant-parameter fast paths, and sampling.
* [Design notes](design.md): the *why* behind those choices and the questions still open, one decision per section,
    with the trade-offs spelled out.

If instead you want to build, test, or extend the project, the practical entry point is
[Contributing](../contributing.md).
