---
icon: lucide/table
---

# Column-valued parameters

Any distribution parameter accepts a scalar, a column name (`str`), a `pl.Expr`, or a `pl.Series`,
and you can mix them freely:

```python
import polars as pl
import polars_stats as ps

ps.Normal(mean=pl.col("mu"), std_dev=1.0)  # column mean, scalar scale
ps.LogNormal(mu=0.0, sigma=1.0)  # all scalar
ps.Uniform(min="min", max="max")  # column names as strings
ps.Binomial(n=pl.col("size"), p=pl.col("probas"))  # all expressions
```

A column-parameterised instance evaluates one distribution per row, vectorised and lazy: every row can carry its own
parameters, and the same method call scores each row against its own distribution.

```python exec="yes" source="above" session="column-parameters" result="python"
import polars as pl
import polars_stats as ps

df = pl.DataFrame(
    {
        "mu": [0.0, 10.0],
        "sigma": [1.0, 2.0],
        "x": [0.5, 11.2],
    }
)

result = df.with_columns(p=ps.Normal(mean="mu", std_dev="sigma").pdf("x"))
print(result)
```

Method arguments follow the same rule: `pdf("x")` reads column `x`, exactly like `pdf(pl.col("x"))`.
A bare string is always a column reference, never a literal.

## Why this is the point

`scipy.stats` and `numpy` parameterise a distribution once, then evaluate it at an array of points. None of them express
*"the density of `x` under a Normal whose mean is column `mu` and standard deviation is column `sigma`"*
without a Python loop or a per-group `apply`.

That row-varying, vectorised, lazy-native case is what `polars-stats` targets; see [Lazy pipelines](lazy-pipelines.md)
for how it composes inside a query.
