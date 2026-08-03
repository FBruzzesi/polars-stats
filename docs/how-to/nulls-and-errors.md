---
icon: lucide/triangle-alert
---

# Handle nulls and errors

One rule governs the whole surface: **a null input gives a null result; an invalid parameter raises.** This page shows
how to work with that. For the exhaustive table of cases, see
[Reference / Parameters and contracts](../reference/parameters-and-contracts.md#nulls-nans-and-errors).

## Let nulls flow through

A `null` in any input on a row produces `null` on that row, never a stand-in constant, so you can score a frame with
gaps and deal with them downstream:

```python exec="yes" source="above" session="nulls-and-errors" result="python"
import polars as pl
import polars_stats as ps

df = pl.DataFrame({"x": [0.0, None, 1.0]}, schema={"x": pl.Float64})

print(df.with_columns(density=ps.Normal().pdf("x")))
```

A null *parameter* behaves the same way: that row's result is null, and no error is raised. This matters when
parameters are estimated, because an under-sized group yields a null `sigma`.

## Find the rows that would raise

An invalid parameter *value* (`sigma <= 0`, `max <= min`, `p` outside `[0, 1]`) fails the whole evaluation with a
`ComputeError`. Locate the offending rows with an ordinary filter before scoring:

```python exec="yes" source="above" session="nulls-and-errors" result="python"
frame = pl.DataFrame(
    {
        "reading": [1.0, 2.0, 3.0],
        "mu": [0.0, 0.0, 0.0],
        "sigma": [1.0, -1.0, 0.0],
    }
)

print(frame.filter(pl.col("sigma") <= 0))
```

## Score anyway, quarantining the bad rows

Filter the invalid rows out of the scored branch. Nulls need no special handling; they propagate:

```python exec="yes" source="above" session="nulls-and-errors" result="python"
dist = ps.Normal(mu="mu", sigma="sigma")

scored = frame.filter(pl.col("sigma") > 0).with_columns(upper_tail=dist.sf("reading"))
quarantined = frame.filter(pl.col("sigma") <= 0)

print(scored)
print(quarantined)
```

If you would rather keep every row, replace the invalid parameters with `null` and let the result null out:

```python exec="yes" source="above" session="nulls-and-errors" result="python"
guarded = frame.with_columns(
    sigma=pl.when(pl.col("sigma") > 0).then("sigma").otherwise(None)
).with_columns(upper_tail=dist.sf("reading"))
print(guarded)
```

Doing this converts a loud failure into a silent one, so do it only where a null result is genuinely the answer you
want.

## Catch the error instead

When failing the query is acceptable and you only need to report it, catch `pl.exceptions.ComputeError`. The message
names the parameter and the offending values:

```python exec="yes" source="above" session="nulls-and-errors" result="python"
try:
    frame.with_columns(dist.sf("reading"))
except pl.exceptions.ComputeError as exc:
    print(str(exc).splitlines()[0])
```

Scalar parameters are validated once and column parameters per row, but a bad scalar and a bad column row surface the
same way:

```python exec="yes" source="above" session="nulls-and-errors" result="python"
try:
    pl.DataFrame({"x": [0.5]}).with_columns(ps.Normal(mu=0.0, sigma=-1.0).pdf("x"))
except pl.exceptions.ComputeError as exc:
    print(str(exc).splitlines()[0])
```

A wrong parameter *type* is caught earlier, at construction, with a `TypeError`: no query runs.

## Related

* [Reference / Parameters and contracts](../reference/parameters-and-contracts.md#nulls-nans-and-errors): the full
    table, including `NaN`, out-of-support, and out-of-range quantiles.
* [Explanation / Design notes](../explanation/design.md#invalid-parameters-raise-they-never-silently-null): why an
    invalid parameter raises rather than nulling.
