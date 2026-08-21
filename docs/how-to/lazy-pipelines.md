---
icon: lucide/workflow
---

# Compose in lazy pipelines

Keep the statistics inside the query: no `collect()` in the middle, no `to_numpy()`, no Python callback.

## Score inside a lazy query

Every method returns a `pl.Expr`, so it slots into a `LazyFrame` plan and runs under the optimiser. Join the
parameters, score, filter, and collect once:

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
    .with_columns(score=ps.Normal(mu="mu", sigma="sigma").sf("reading"))
    .filter(pl.col("score") < 0.01)
    .collect()
)
print(anomalies)
```

Each reading is scored against the baseline carried on its own row, the tail probability is filtered in the same plan,
and the whole thing collects once.

## Aggregate a density within a window

Methods are elementwise, so they compose inside a window expression: compute per row, then aggregate with `over`. Here
each row's density is divided by its group's total to give a within-group share, kept aligned to the frame:

```python exec="yes" source="above" session="lazy-pipelines" result="python"
df = pl.DataFrame(
    {
        "group": ["a", "a", "b"],
        "mu": [0.0, 0.0, 5.0],
        "sigma": [1.0, 1.0, 2.0],
        "x": [0.0, 1.0, 5.0],
    }
)
dist = ps.Normal(mu="mu", sigma="sigma")
print(
    df.with_columns(
        density=dist.pdf("x"),
        group_share=dist.pdf("x") / dist.pdf("x").sum().over("group"),
    )
)
```

## Collect per-group values with `group_by`

Under `group_by(...).agg(...)` the same elementwise call collects into one list per group rather than reducing:

```python exec="yes" source="above" session="lazy-pipelines" result="python"
print(df.group_by("group", maintain_order=True).agg(density=dist.pdf("x")))
```

Wrap it in an aggregation if you want a single number per group, for example
`agg(mean_density=dist.pdf("x").mean())`.

## Related

* [Use column-valued parameters](column-parameters.md): where those `mu` / `sigma` columns come from.
* [Explanation / Architecture](../explanation/architecture.md#column-valued-parameters): why a scalar parameter is a
    length-1 literal, and how Rust broadcasts it so these methods stay elementwise under partitioning.
