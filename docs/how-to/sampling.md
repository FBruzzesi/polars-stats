---
icon: lucide/dices
---

# Sample reproducibly

Draw random variates as part of a query, with results you can reproduce on another machine.

## Draw one variate per row

`sample(seed)` draws from each row's own distribution:

```python exec="yes" source="above" session="sampling" result="python"
import polars as pl
import polars_stats as ps

dist = ps.Normal(mu="mu", sigma="sigma")
df = pl.DataFrame({"mu": [0.0, 10.0], "sigma": [1.0, 2.0]})

print(df.with_columns(draw=dist.sample(seed=42)))
```

## Draw several variates per row

`samples(size, seed)` returns an `Array` of width `size` per row:

```python exec="yes" source="above" session="sampling" result="python"
print(df.with_columns(draws=dist.samples(3, seed=42)))
```

`samples(size=1)` matches `sample` bit for bit for the same seed, and increasing `size` extends each row's array
without changing the draws already there.

## Reduce or expand the draws

Stay in the array to summarise per row:

```python exec="yes" source="above" session="sampling" result="python"
print(
    df.with_columns(draws=dist.samples(4, seed=7)).with_columns(
        mc_mean=pl.col("draws").arr.mean(),
        mc_max=pl.col("draws").arr.max(),
    )
)
```

Or `explode` to get one row per draw:

```python exec="yes" source="above" session="sampling" result="python"
print(df.with_columns(draw=dist.samples(2, seed=7)).explode("draw"))
```

## Make results reproducible

Pass an integer `seed`. The output is then deterministic across operating systems, architectures, chunk layouts, thread
counts, and both the in-memory and streaming engines:

```python exec="yes" source="above" session="sampling" result="python"
lazy = pl.LazyFrame({"mu": [0.0, 10.0], "sigma": [1.0, 2.0]})
expr = ps.Normal(mu="mu", sigma="sigma").sample(seed=42)

print(lazy.with_columns(draw=expr).collect())
print(lazy.with_columns(draw=expr).collect(engine="streaming"))
```

Omit `seed` (or pass `None`) for non-reproducible draws seeded from OS entropy. There is no global seed to set: the
`seed` argument is the only control, which means a sampled column is reproducible from the query alone.

Reproducibility holds **across releases** too, from `0.0.1` onward: for a given seed, row position and parameters,
`sample` and `samples` return the same values on every later version. Changing a draw algorithm is therefore a
breaking change, and it will be called out as one in that version's release notes rather than shipped alongside an
unrelated fix.

Determinism does not depend on evaluation order. Each row derives its own generator from `(seed, row_index)`, so a
row's draw depends only on its position in the frame, never on how Polars chunked or threaded the data.

## Related

* [Reference / Parameters and contracts](../reference/parameters-and-contracts.md#sampling): the element dtype per
    distribution, output length rules, and seed behaviour.
* [Explanation / Architecture](../explanation/architecture.md#sampling): the per-row RNG mechanics.
* [Explanation / Design notes](../explanation/design.md#sampling-derives-a-fresh-per-row-rng-from-root_seed-row_index):
    why it is built that way, and the design that was replaced.
