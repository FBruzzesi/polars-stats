---
icon: lucide/table
---

# Use column-valued parameters

Give each row its own distribution, either from columns you already have or from parameters you compute in the same
query.

## Pass columns as parameters

Any parameter accepts a scalar, a column name (`str`), a `pl.Expr`, or a `pl.Series`, and you can mix them in one
constructor:

```python
import polars as pl
import polars_stats as ps

ps.Normal(mu=pl.col("mu"), sigma=1.0)  # column mean, scalar scale
ps.Normal(mu=0.0, sigma=1.0)  # all scalar
ps.Uniform(min="min", max="max")  # column names as strings
ps.Binomial(n=pl.col("size"), p=pl.col("probas"))  # all expressions
```

The instance then evaluates one distribution per row:

```python exec="yes" source="above" session="column-parameters" result="python"
import polars as pl
import polars_stats as ps

df = pl.DataFrame({"mu": [0.0, 10.0], "sigma": [1.0, 2.0], "x": [0.5, 11.2]})

print(df.with_columns(p=ps.Normal(mu="mu", sigma="sigma").pdf("x")))
```

Method arguments follow the same rule: `pdf("x")` reads column `x`, exactly like `pdf(pl.col("x"))`. A bare string is
always a column reference, never a literal.

!!! warning "Floats are strict"

    A float parameter rejects an `int`: `ps.Normal(mu=0, sigma=1)` raises `TypeError`. Write `0.0` and `1.0`.
    Count parameters are the mirror image (`Binomial(n=10, p=0.3)` wants an `int` for `n`). The full table is in
    [Reference / Parameters and contracts](../reference/parameters-and-contracts.md#accepted-inputs).

## Keep parameters in a lookup table

When parameters belong to an entity rather than a row, join them in and pass the joined columns:

```python exec="yes" source="above" session="column-parameters" result="python"
readings = pl.DataFrame(
    {
        "sensor": ["a", "a", "b", "b"],
        "reading": [9.8, 10.4, 100.0, 135.0],
    }
)
baselines = pl.DataFrame(
    {"sensor": ["a", "b"], "mu": [10.0, 100.0], "sigma": [0.5, 2.0]}
)

print(
    readings.join(baselines, on="sensor", maintain_order="left").with_columns(
        upper_tail=ps.Normal(mu="mu", sigma="sigma").sf("reading")
    )
)
```

## Derive parameters from the data

There is no `fit` method. Estimate the parameters with ordinary Polars expressions, then feed those columns in. A
per-group Normal fit is a mean and a standard deviation:

```python exec="yes" source="above" session="column-parameters" result="python"
history = pl.DataFrame(
    {
        "sensor": ["a", "a", "a", "a", "b", "b", "b", "b"],
        "reading": [1.0, 1.2, 0.9, 1.1, 10.0, 10.5, 9.5, 10.2],
    }
)

print(
    history.with_columns(
        mu=pl.col("reading").mean().over("sensor"),
        sigma=pl.col("reading").std().over("sensor"),
    ).with_columns(upper_tail=ps.Normal(mu="mu", sigma="sigma").sf("reading"))
)
```

The same shape works for a rolling baseline (`rolling_mean` / `rolling_std` instead of `mean` / `std`) or for
method-of-moments estimates of other distributions
(`Exponential(rate=1 / pl.col("reading").mean().over("sensor"))`).

Estimated parameters can be invalid: a group with one row gives `sigma = null`, and that propagates to `null` rather
than raising. See [Handle nulls and errors](nulls-and-errors.md).

## Related

* [Compose in lazy pipelines](lazy-pipelines.md): the same parameters inside a `LazyFrame`, a window, or a `group_by`.
* [Reference / Parameters and contracts](../reference/parameters-and-contracts.md): every accepted input type.
* [Why polars-stats](../explanation/index.md): why row-varying parameters are the point.
