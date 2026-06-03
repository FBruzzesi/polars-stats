---
icon: lucide/sigma
---

# polars-stats

`polars-stats` is a [Polars](https://pola.rs) expression plugin that exposes
[`scipy.stats`](https://docs.scipy.org/doc/scipy/reference/stats.html)-style probability distributions natively inside
Polars expressions, with two properties that `scipy` and `numpy` do not give you:

* **Column-valued parameters**: any distribution parameter can be a scalar *or* a Polars expression. A single instance
  describes a different distribution per row:

    ```python
    Normal(mean=pl.col("mu"), std_dev=pl.col("sigma")).cdf(pl.col("x"))
    ```

* **Lazy-compatible**: every method returns a `pl.Expr`, so it composes inside a `LazyFrame` query under the
    optimiser, with no materialisation. Sampling has one caveat, covered in [Sampling](user-guide/sampling.md).

The math runs in Rust on top of the [`statrs`](https://docs.rs/statrs) crate; the Python layer is a thin, typed surface
of distribution classes.

## Why

Statistical work in Polars today means falling back to `.to_pandas()` / `.to_numpy()`, then reaching for one of:

* `scipy.stats`, which exits the lazy engine and materialises everything,
* Python UDFs via `map_elements`, which are slow and hold the GIL,
* hand-rolled per-distribution expressions, which are ad hoc and error-prone.

None of them express *"the PDF of `x` under a Normal whose mean is column `mu` and standard deviation is column `sigma`"*.

This row-varying, vectorised, lazy-native case is what `polars-stats` targets.

## Install

!!! warning "Pre-release"

    `polars-stats` is alpha software. The public API may change.

```bash
pip install polars-stats
```

The only runtime dependency is `polars>=1.15`. Wheels ship for Linux, macOS, and Windows; see
[Contributing](contributing.md) to build from source.

## Quick example

The flagship use case, per-row distribution parameters for anomaly scoring:

```python exec="yes" source="above" session="index" result="python"
import polars as pl
from polars_stats import Normal

readings = pl.LazyFrame(
    {
        "value": [9.8, 101.0, 12.1, 250.0],
        "baseline_mu": [10.0, 100.0, 10.0, 100.0],
        "baseline_sigma": [0.5, 2.0, 0.5, 2.0],
    }
)

norm = Normal(mean="baseline_mu", std_dev="baseline_sigma")

anomalies = (
    readings.with_columns(upper_tail=norm.sf("value"))
    .filter(pl.col("upper_tail") < 0.01)
    .collect()
)
print(anomalies)
```

Each row is scored against its own `Normal(baseline_mu, baseline_sigma)`, in one vectorised pass, without leaving the
lazy engine.

## Where to next

* [Getting started](getting-started.md): install, build, and the core usage patterns.
* [API reference](reference/index.md): the catalogue, the method surface, and worked examples, with the generated
  docstrings split into [Continuous](reference/continuous.md) and [Discrete](reference/discrete.md).
* [Architecture](architecture.md) and [Design notes](design.md): how it is wired and why.

## License

MIT.
