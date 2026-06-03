---
icon: lucide/rocket
---

# Getting started

## Install

```bash
pip install polars-stats
```

Runtime needs only `polars >= 1.15` and Python `>= 3.10`. To build from source, see [Contributing](contributing.md).

## Construct a distribution

Import a class and parameterise it. Each class names its parameters after the distribution's conventional parameters
(`mean` / `std_dev`, `min` / `max`, ...); the class docstring gives the `scipy.stats` equivalent.

```python exec="yes" source="above" session="getting-started"
import polars as pl
from polars_stats import Normal

dist = Normal(mean=0.0, std_dev=1.0)
```

## Evaluate a method

Every method returns a `pl.Expr`, so it composes with the rest of your query. A method argument can be a column name
(`str`), a `pl.Expr`, or a scalar:

```python exec="yes" source="above" session="getting-started" result="python"
df = pl.DataFrame({"x": [-1.0, 0.0, 1.0]})

result = df.with_columns(
    density=dist.pdf("x"),
    upper_tail=dist.sf("x"),
)
print(result)
```

The full method surface (`pdf`/`pmf`, `cdf`, `sf`, `ppf`, the `log_*` family, moments, sampling) is listed in
[Distributions](distributions.md#method-surface).

## Where to next

The defining feature is that parameters themselves can be columns, so one instance describes a different distribution
per row:

* [Column-valued parameters](user-guide/column-parameters.md): the differentiator, with the accepted parameter types.
* [Lazy pipelines](user-guide/lazy-pipelines.md): composing methods inside `LazyFrame` queries, `over` / `group_by`.
* [Sampling](user-guide/sampling.md): `sample` / `samples`, seeding, reproducibility.
* [Nulls and errors](user-guide/nulls-and-errors.md): the null-propagation and invalid-parameter contract.
