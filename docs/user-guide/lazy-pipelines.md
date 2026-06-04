---
icon: lucide/workflow
---

# Lazy pipelines

Every method returns a `pl.Expr`, so it slots into a `LazyFrame` pipeline under the query optimiser, with no
materialisation and no exit from the engine. A per-row anomaly score is one `with_columns` away.

```python exec="yes" source="above" session="lazy-pipelines" result="python"
import polars as pl
import polars_stats as ps

readings = pl.LazyFrame(
    {
        "sensor": ["a", "a", "b", "b"],
        "reading": [9.8, 10.4, 100.0, 135.0],
    }
)

baselines = pl.LazyFrame(
    {
        "sensor": ["a", "b"],
        "mu": [10.0, 100.0],
        "sigma": [0.5, 2.0],
    }
)

anomalies = (
    readings.join(baselines, on="sensor")
    .with_columns(score=ps.Normal(mean="mu", std_dev="sigma").sf("reading"))
    .filter(pl.col("score") < 0.01)
    .collect()
)
print(anomalies)
```

Each reading is scored against the baseline distribution carried on its own row, the upper-tail probability is filtered
in the same lazy plan, and the whole thing collects once.

## `over` and `group_by`

Methods are elementwise by contract, so they behave correctly under `over` and `group_by`: the expression is invoked
once per partition, not as an aggregation. A distribution parameterised per group scores each row within its group.

Because the density is elementwise, it composes inside a window expression: compute it per row, then aggregate within
each group with `over`. Here each row's density is divided by its group's total to give a within-group share, kept
aligned to the frame:

```python exec="yes" source="above" session="lazy-pipelines" result="python"
df = pl.DataFrame(
    {
        "group": ["a", "a", "b"],
        "mu": [0.0, 0.0, 5.0],
        "sigma": [1.0, 1.0, 2.0],
        "x": [0.0, 1.0, 5.0],
    }
)
dist = ps.Normal(mean="mu", std_dev="sigma")
print(
    df.with_columns(
        density=dist.pdf("x"),
        group_share=dist.pdf("x") / dist.pdf("x").sum().over("group"),
    )
)
```

Under `group_by(...).agg(...)` the same elementwise call collects into one list per group rather than reducing:

```python exec="yes" source="above" session="lazy-pipelines" result="python"
print(df.group_by("group", maintain_order=True).agg(density=dist.pdf("x")))
```

This is what the scalar-to-column coercion buys: a Python `float` parameter is expanded to a row-aligned expression, so
the plugin always receives a length-matched input and stays elementwise under partitioning. See
[Architecture / Column-valued parameters](../architecture.md#column-valued-parameters) for the mechanics.
