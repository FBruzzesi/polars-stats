---
icon: lucide/map
---

# User guide

Task-oriented guides for the patterns you will actually write. Each page is self-contained and runnable; if you have
not installed the package yet, start with [Getting started](../getting-started.md).

* [Column-valued parameters](column-parameters.md): any parameter can be a column, so a single instance describes
    a different distribution per row.
* [Lazy pipelines](lazy-pipelines.md): every method returns a `pl.Expr`, so it composes inside `LazyFrame` queries
    under the optimiser, including `over` / `group_by`.
* [Sampling](sampling.md): the `sample` / `samples` contract, seeding, and why results are reproducible regardless of
    chunking and threading.
* [Nulls and errors](nulls-and-errors.md): null inputs propagate, invalid parameters raise, and where the line sits.

For the full catalogue and method surface, see the [API reference](../reference/index.md).
