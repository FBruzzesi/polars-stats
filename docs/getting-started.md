---
icon: lucide/rocket
---

# Getting started

The 60-second version: install, evaluate one distribution, know where to go next. For a guided build of something
real, take the [tutorial](tutorial.md) instead.

## Install

```bash
pip install polars-stats
```

Requires Python `>=3.10` and `polars>=1.15`. On the platforms below there is nothing to compile: the wheels are
`abi3`, so a single wheel per platform serves every Python from 3.10 up.

| Platform | Targets |
| --- | --- |
| Linux (glibc, manylinux 2.17) | `x86_64`, `aarch64` |
| macOS | `x86_64`, `arm64` (Apple silicon) |
| Windows | `x64` |
| Source | sdist, built on install and needs a Rust toolchain |

Alpine (musl) and Windows on ARM are the two gaps: `polars` itself publishes wheels for both, `polars-stats` does
not yet, so `pip` falls back to the sdist there. [Open an issue](https://github.com/FBruzzesi/polars-stats/issues)
if you need one and it gets added.

To build from source deliberately, see [Contributing](contributing.md#build-from-source).

## Evaluate a distribution

Pick a class, parameterise it, call a method. Every method returns a `pl.Expr`, so it goes wherever an expression
goes:

```python exec="yes" source="above" session="getting-started" result="python"
import polars as pl
import polars_stats as ps

dist = ps.Normal(mu=0.0, sigma=1.0)

df = pl.DataFrame({"x": [-1.0, 0.0, 1.0]})

print(
    df.with_columns(
        density=dist.pdf("x"),
        upper_tail=dist.sf("x"),
    )
)
```

Parameters are named after the distribution's own convention (`mu` / `sigma`, `min` / `max`, `a` / `b`), and each
class docstring gives the `scipy.stats` translation. A `str` argument such as `pdf("x")` is a column reference,
identical to `pdf(pl.col("x"))`.

## Parameters can be columns

Any parameter accepts a column, so one instance describes a different distribution per row:

```python exec="yes" source="above" session="getting-started" result="python"
per_row = pl.DataFrame({"mu": [0.0, 10.0], "sigma": [1.0, 2.0], "x": [0.5, 11.2]})

print(per_row.with_columns(density=ps.Normal(mu="mu", sigma="sigma").pdf("x")))
```

Each row got its own `Normal`, and the density came back as an ordinary column you can filter, join, or aggregate on.

## Where to next

* [Tutorial](tutorial.md): build a per-row anomaly detector end to end, in seven steps.
* [How-to guides](how-to/index.md): recipes for parameters from columns, lazy queries, sampling, nulls, and
    migrating from `scipy.stats`.
* [API reference](reference/index.md): the distribution catalogue, the method surface, and the input contracts.
* [Explanation](explanation/index.md): why the library exists and how it is built.
