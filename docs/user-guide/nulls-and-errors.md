---
icon: lucide/triangle-alert
---

# Nulls and errors

One rule governs the whole surface: **an invalid parameter is a modelling error, raised loudly; `null` is reserved for
`null` inputs.** A silent `null` from a bad parameter is indistinguishable from a legitimately-null input and would
propagate wrong answers downstream.

## Null inputs propagate

A `null` in any input on a row produces `null` on that row, never a stand-in constant:

```python exec="yes" source="above" session="nulls-and-errors" result="python"
import polars as pl
import polars_stats as ps

df = pl.DataFrame({"x": [0.0, None, 1.0]}, schema={"x": pl.Float64})

print(df.with_columns(density=ps.Normal().pdf("x")))
```

## Invalid parameters raise

An invalid parameter *value* (e.g. `std_dev <= 0`) raises a `ComputeError` at evaluation; it never silently nulls.
Scalars are coerced to columns and validated per row, so a bad scalar and a bad column row surface identically.

```python exec="yes" source="above" session="nulls-and-errors" result="python"
df = pl.DataFrame({"x": [0.5]})

try:
    df.with_columns(ps.Normal(mean=0.0, std_dev=-1.0).pdf("x"))
except pl.exceptions.ComputeError as exc:
    print(str(exc).splitlines()[0])
```

A wrong parameter *type* (e.g. passing a `list`) is rejected earlier, at construction, with a `TypeError`.

## Full contract

| Situation | When detected | Behaviour |
|---|---|---|
| Wrong parameter *type* at construction (e.g. a `list`) | Python `__init__` | `TypeError` |
| Invalid parameter *value*, scalar or column row (`std_dev <= 0`, `max <= min`, `p` outside `[0, 1]`) | Rust evaluation | `ComputeError`, fails the whole evaluation, never silently nulls |
| Any input `null` on a row | per row | `null` on that row |
| `q` outside `[0, 1]` in `ppf` | per row | `null` |
| `x` outside the support (e.g. `pdf` below a Uniform's `min`) | per row | `0.0` (matches `scipy`) |
| `pmf(3.5)` for a discrete distribution | per row | `0.0` (matches `scipy`) |
| Type mismatch (a string column into `pdf`) | Rust evaluation | `ComputeError` |

Every distribution shipped today (`Normal`, `LogNormal`, `Uniform`, `Bernoulli`, `Binomial`) has finite moments on its valid
parameter range, so this contract is exhaustive for them. The policy for distributions whose moments can be undefined
(some are on the roadmap) is set out in [Design notes](../design.md#moments-that-are-undefined).

Raising is loud, uniform across distributions, and uniform across scalar vs column inputs. The rationale is in
[Design notes](../design.md#invalid-parameters-raise-they-never-silently-null).
