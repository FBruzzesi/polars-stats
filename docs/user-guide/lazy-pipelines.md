---
icon: lucide/workflow
---

# Lazy pipelines

Every method returns a `pl.Expr`, so it slots into a `LazyFrame` pipeline under the query optimiser, with no
materialisation and no exit from the engine. A per-row anomaly score is one `with_columns` away.

```python exec="yes" source="above" session="lazy-pipelines" result="python"
import polars as pl
from polars_stats import Normal

readings = pl.LazyFrame(
    {
        "sensor": ["a", "a", "b", "b"],
        "reading": [9.8, 10.4, 100.0, 250.0],
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
    .with_columns(score=Normal(mean="mu", std_dev="sigma").sf("reading"))
    .filter(pl.col("score") < 0.01)
    .collect()
)
print(anomalies)
```

Each reading is scored against the baseline distribution carried on its own row, the upper-tail probability is filtered
in the same lazy plan, and the whole thing collects once.

## `over` and `group_by`

Methods are elementwise by contract, so they behave correctly under `over` and `group_by`: the expression is invoked
once per partition, not as an aggregation. A distribution parameterised per group scores each row within its group:

```python exec="yes" source="above" session="lazy-pipelines" result="python"
df = pl.DataFrame(
    {
        "group": ["a", "a", "b"],
        "mu": [0.0, 0.0, 5.0],
        "sigma": [1.0, 1.0, 2.0],
        "x": [0.0, 1.0, 5.0],
    }
)

print(
    df.with_columns(density=Normal(mean="mu", std_dev="sigma").pdf("x").over("group"))
)
```

This is what the scalar-to-column coercion buys: a Python `float` parameter is expanded to a row-aligned expression, so
the plugin always receives a length-matched input and stays elementwise under partitioning. See
[Architecture / Column-valued parameters](../architecture.md#column-valued-parameters) for the mechanics.
