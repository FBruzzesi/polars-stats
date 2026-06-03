---
icon: lucide/dices
---

# Sampling

`sample` draws one variate per row from that row's distribution; `samples(size)` draws a fixed-width `Array` per row.

```python exec="yes" source="above" session="sampling" result="python"
import polars as pl
from polars_stats import Normal

df = pl.DataFrame({"mu": [0.0, 10.0], "sigma": [1.0, 2.0]})

print(df.with_columns(draw=Normal(mean="mu", std_dev="sigma").sample(seed=42)))
```

`samples(size, seed)` returns an `Array(inner=..., shape=size)` per row, drawing `size` independent variates:

```python exec="yes" source="above" session="sampling" result="python"
print(df.with_columns(draws=Normal(mean="mu", std_dev="sigma").samples(3, seed=42)))
```

## Output length and dtype

`sample` returns a `pl.Expr` of one draw per row. Output length follows the surrounding context: the frame length under
`select` / `with_columns`, the partition length under `over` / `group_by`.

The sample element dtype is per distribution and is not normalised to `Float64`:

| Distribution | Sample dtype |
|---|---|
| `Bernoulli` | `Boolean` |
| `Normal`, `LogNormal`, `Uniform` | `Float64` |
| other discrete | `UInt64` |

## Seeding and reproducibility

Pass an integer `seed` for output that is deterministic across platforms, chunking, thread count, and execution engine:

| `seed` | Behaviour |
|---|---|
| `int` | Deterministic across OS, architecture, chunking, thread count, and engine (in-memory or streaming) |
| `None` | Non-reproducible by design (OS entropy) |

Determinism comes from deriving a fresh per-row generator from `(seed, row_index)` rather than advancing one shared RNG
in iteration order. That keying makes a row's draw depend only on its global position, not on how Polars chunks,
threads, or morsels the data, so `sample` is genuinely elementwise and correct under `over` / `group_by`. The mechanics
are in [Architecture / Sampling](../architecture.md#sampling).

!!! info "Streaming engine"

    Seeded sampling is engine-invariant: a `LazyFrame` collected with `engine="streaming"` yields the same draws as
    the default in-memory engine. A property test (`tests/property/sample_test.py`) asserts this across a multi-chunk
    source for every distribution. Note that the row index is a whole-frame quantity, so the sampling step is not
    memory-bounded streaming, it is correct, not lazy in the streaming sense.
