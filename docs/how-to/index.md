---
icon: lucide/map
---

# How-to guides

Recipes for specific problems. Each page assumes you have `polars-stats` installed and know what a distribution is; if
not, start with [Getting started](../getting-started.md) or the [tutorial](../tutorial.md).

| I want to ... | Guide |
|---|---|
| give every row its own distribution parameters, or derive them from the data | [Use column-valued parameters](column-parameters.md) |
| put a distribution inside a `LazyFrame` query, a window, or a `group_by` | [Compose in lazy pipelines](lazy-pipelines.md) |
| draw random variates that are reproducible across runs and machines | [Sample reproducibly](sampling.md) |
| decide what happens to null inputs and bad parameters in my pipeline | [Handle nulls and errors](nulls-and-errors.md) |
| port existing `scipy.stats` code | [Migrate from scipy.stats](migrate-from-scipy.md) |

For the exact set of accepted inputs and returned dtypes, see
[Reference / Parameters and contracts](../reference/parameters-and-contracts.md). For why the library works this way,
see [Explanation](../explanation/index.md).
