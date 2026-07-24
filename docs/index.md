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
    import polars as pl
    import polars_stats as ps

    norm = ps.Normal(mu=pl.col("mu"), sigma=pl.col("sigma"))
    norm.cdf(pl.col("x"))
    ```

* **Lazy-compatible**: every method returns a `pl.Expr`, so it composes inside a `LazyFrame` query under the
    optimiser, with no materialisation.

The math runs in Rust on top of the [`statrs`](https://docs.rs/statrs) crate; the Python layer is a thin, typed surface
of distribution classes.

## Why

Statistical work in Polars today means falling back to `.to_pandas()` / `.to_numpy()`, then reaching for one of:

* `scipy.stats`, which exits the lazy engine and materialises everything,
* Python UDFs via `map_elements`, which are slow and hold the GIL,
* hand-rolled per-distribution expressions, which are ad hoc and error-prone.

None of them express *"the PDF of `x` under a Normal whose mean is column `mu` and standard deviation is column `sigma`"*.

This row-varying, vectorised, lazy-native case is what `polars-stats` targets.

## Quick example

The flagship use case, per-row distribution parameters for anomaly scoring:

```python exec="yes" source="above" session="index" result="python"
import polars as pl
import polars_stats as ps

readings = pl.LazyFrame(
    {
        "value": [9.8, 101.0, 12.1, 250.0],
        "mu": [10.0, 100.0, 10.0, 100.0],
        "sigma": [0.5, 2.0, 0.5, 2.0],
    }
)

norm = ps.Normal(mu="mu", sigma="sigma")

anomalies = (
    readings.with_columns(upper_tail=norm.sf("value"))
    .filter(pl.col("upper_tail") < 0.01)
    .collect()
)
print(anomalies)
```

Each row is scored against its own `Normal(mu, sigma)`, in one vectorised pass, without leaving the lazy engine.

## Installation

```bash
pip install polars-stats
```

Runtime needs `polars>=1.15` and Python `>=3.10`

## A note on the Rust code

**I am not a Rust expert, and a good part of the Rust layer was written with AI assistance.**

What I vouch for is the behaviour, which is pinned by an extensive test suite: parity against `scipy.stats` on every
method, property-based invariants, and bit-identity between the constant-parameter fast paths and the general
per-row paths.

Treat the Rust idioms with the appropriate skepticism: if you spot something that should be written differently,
an issue or PR is very welcome.

## Related projects

`polars-stats` is not the first take on statistics inside Polars expressions. Two projects cover neighbouring
ground, and if your need matches their scope they may serve you well:

* [`polars-random`](https://github.com/diegoglozano/polars-random) generates random columns as native Polars
  expressions (uniform, normal, binomial, integers), with column-valued parameters and seeding. Its focus is
  sampling; `polars-stats` treats sampling as one method of a full distribution object, next to `pdf` / `cdf` /
  `sf` / `ppf`, their numerically stable log variants, and closed-form moments.
* [`polars_normal_stats`](https://github.com/MaxwellB13/polars_normal_stats) exposes `normal_cdf` / `normal_ppf` /
  `normal_pdf` as expressions, also with per-row parameters. It covers the Normal distribution through three
  functions; `polars-stats` generalises the same idea to a catalogue of distributions behind one scipy-like class
  API, and adds survival functions, `log_cdf` / `log_sf`, and reproducible sampling.

## Where to next

* [Getting started](getting-started.md): the core usage patterns.
* [API reference](reference/index.md): the catalogue, the method surface, and worked examples, with the generated
  docstrings split into [Continuous](reference/continuous.md) and [Discrete](reference/discrete.md).
* [Architecture](explanation/architecture.md) and [Design notes](explanation/design.md): how it is wired and why.

## License

This project is licensed under the MIT license.
