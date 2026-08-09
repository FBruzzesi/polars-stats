---
icon: lucide/sigma
---

# polars-stats

`polars-stats` is a [Polars](https://pola.rs) expression plugin that exposes
[`scipy.stats`](https://docs.scipy.org/doc/scipy/reference/stats.html)-style probability distributions natively inside
Polars expressions:

* **Lazy-native**: every method returns a `pl.Expr`, so a distribution composes inside a `LazyFrame` query under the
  optimiser, with no materialisation.

* **Column-valued parameters**: any distribution parameter can be a scalar *or* a Polars expression. A single instance
  describes a different distribution per row:

    ```python
    import polars as pl
    import polars_stats as ps

    norm = ps.Normal(mu=pl.col("mu"), sigma=pl.col("sigma"))
    norm.cdf(pl.col("x"))
    ```

* **Polars null and error semantics**: a `null` input gives a `null` result, and an invalid parameter raises a
  `ComputeError` rather than silently returning `NaN`.

* **Reproducible sampling**: every draw is keyed on `(seed, row index)`, so a seeded column repeats across runs,
  chunkings, thread counts, and both engines.

`scipy` already does the per-row maths: `stats.norm(loc=mu_array, scale=sigma_array).sf(x_array)` broadcasts parameter
arrays and scores every element against its own distribution, vectorised, with no Python loop. The difference is where
the result lands. `scipy` returns a NumPy array, so a `LazyFrame` has to `collect()` first, pushdown stops at that
boundary, and realigning the result through later joins and filters is your problem. Here it stays a `pl.Expr` the
planner can see. [Why polars-stats](explanation/index.md) has the full comparison.

The math runs in Rust on top of the [`statrs`](https://docs.rs/statrs) crate; the Python layer is a thin, typed surface
of distribution classes. [Numerical accuracy](explanation/accuracy.md) covers how that is checked and where the known
limits are.

## Quick example

Anomaly scoring, where each row carries its own baseline:

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

Runtime needs `polars>=1.15` and Python `>=3.10`.

## Documentation

The docs follow the [Diátaxis](https://diataxis.fr/) split, so pick the entry point that matches what you are doing:

| Page | What it covers |
|---|---|
| **[Getting started](getting-started.md)** | install and evaluate your first distribution, in a minute |
| **[Tutorial](tutorial.md)** | learn by building a per-row anomaly detector, step by step |
| **[How-to guides](how-to/index.md)** | recipes: parameters from columns, lazy pipelines, sampling, nulls, porting `scipy` code |
| **[API reference](reference/index.md)** | the catalogue, the method surface, the input and error contracts, generated docstrings |
| **[Explanation](explanation/index.md)** | why the library exists, what it trades away, and how it is built |
| **[Contributing](contributing.md)** | build from source, run the checks, add a distribution |

## A note on the Rust code

**I am not a Rust expert, and a good part of the Rust layer was written with AI assistance.**

What I vouch for is the behaviour, which is pinned by an extensive test suite: parity against `scipy.stats` on every
method, property-based invariants, and bit-identity between the constant-parameter fast paths and the general
per-row paths.

Treat the Rust idioms with the appropriate skepticism: if you spot something that should be written differently,
an issue or PR is very welcome.

## Related projects

`polars-stats` is not the first take on statistics inside Polars expressions. Three projects cover neighbouring
ground, and if your need matches their scope they may serve you well:

* [`polars-random`](https://github.com/diegoglozano/polars-random) generates random columns as native Polars
  expressions (uniform, normal, binomial, integers), with column-valued parameters, per-call seeds, and a global
  `set_random_seed`. It registers `.random` namespaces on `Expr`, `DataFrame`, and `LazyFrame`, which reads very
  naturally when sampling is the whole job. Its focus is sampling; `polars-stats` treats sampling as one method of a
  full distribution object, next to `pdf` / `cdf` / `sf` / `ppf`, their numerically stable log variants, and
  closed-form moments.
* [`polars_rng`](https://github.com/alipatti/polars_rng) exposes one sampling expression per distribution
  (`prng.normal(mu=pl.col("x"), sigma=3)`), also as a Rust plugin over the same `statrs` crate, also with
  column-valued parameters. Its sampling catalogue is wider than what `polars-stats` ships today (Poisson, Gamma,
  Weibull, Laplace, plus categorical and integer draws), so for pure simulation it may be the better fit. The
  differences are scope and reproducibility: it is sampling only, with no `pdf` / `cdf` / `ppf` or moments, and it
  draws from a thread-local RNG with no `seed` argument, where `polars-stats` keys every draw on
  `(seed, row index)` so a seeded column repeats across runs, chunkings, and engines.
* [`polars_normal_stats`](https://github.com/MaxwellB13/polars_normal_stats) covers the Normal distribution through
  three focused expressions, `normal_cdf` / `normal_ppf` / `normal_pdf`, each evaluated at a column of points. Its
  `mean` and `std` travel as plugin kwargs, so they are scalars: the common case, handled in three functions and
  nothing more. `polars-stats` generalises the same idea to a catalogue of distributions behind one scipy-like class
  API, passes parameters as plugin inputs so they can be columns, and adds survival functions, `log_cdf` / `log_sf`,
  and reproducible sampling.

## License

This project is licensed under the MIT license.
