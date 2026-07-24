---
icon: lucide/book-open
---

# API reference

The catalogue, the shared method surface, and worked examples. The full generated docstrings live on two pages:
[Continuous](continuous.md) and [Discrete](discrete.md).

## Catalogue

Shipped today:

| Distribution | Kind | scipy equivalent |
|---|---|---|
| `Beta(a, b)` | continuous | `beta(a, b)` |
| `Exponential(rate)` | continuous | `expon(scale=1 / rate)` |
| `LogNormal(mu, sigma)` | continuous | `lognorm(s=sigma, scale=exp(mu))` |
| `Normal(mu, sigma)` | continuous | `norm(loc=mu, scale=sigma)` |
| `Uniform(min, max)` | continuous | `uniform(loc=min, scale=max - min)` |
| `Bernoulli(p)` | discrete | `bernoulli(p)` |
| `Binomial(n, p)` | discrete | `binom(n, p)` |

Each class names its parameters after the distribution's conventional parameters; the `scipy equivalent` column gives
the `scipy.stats` translation, and each class docstring states it too.

## A continuous distribution end to end

A single instance answers the whole method surface. Here `LogNormal` evaluated at a column of points, with scalar
parameters:

```python exec="yes" source="above" session="reference" result="python"
import polars as pl
import polars_stats as ps

dist = ps.LogNormal(mu=0.0, sigma=1.0)

df = pl.DataFrame({"x": [0.5, 1.0, 2.0]})

print(
    df.with_columns(
        density=dist.pdf("x"),
        cumulative=dist.cdf("x"),
        upper_tail=dist.sf("x"),
        quantile=dist.ppf(0.9),
    )
)
```

Argument-free statistics (`mean`, `median`, `std`, `entropy`, ...) take no point and return one value per row of
parameters. Here each row carries its own `(mu, sigma)`, so a single instance yields the moments of a different
distribution per row:

```python exec="yes" source="above" session="reference" result="python"
params = pl.DataFrame({"mu": [0.0, 1.0], "sigma": [1.0, 0.5]})

per_row = ps.LogNormal(mu="mu", sigma="sigma")

print(
    params.with_columns(
        mean=per_row.mean(),
        median=per_row.median(),
        std=per_row.std(),
        entropy=per_row.entropy(),
    )
)
```

## A discrete distribution

Discrete distributions expose `pmf` instead of `pdf`; everything else is the same surface.

`Bernoulli` samples to a `Boolean` column:

```python exec="yes" source="above" session="reference" result="python"
import polars_stats as ps

coin = ps.Bernoulli(p=0.3)

outcomes = pl.DataFrame({"k": [0, 1]})
print(outcomes.with_columns(mass=coin.pmf("k")))

trials = pl.DataFrame({"trial": range(5)})
print(trials.with_columns(flip=coin.sample(seed=0)))
```

## Method surface

Every distribution exposes the same surface, defined on the base classes. Value-keyed methods take a scalar, a column
name (`str`), or a `pl.Expr`; argument-free statistics take none. All return a `pl.Expr`.

| Method | Continuous | Discrete | Meaning |
|---|---|---|---|
| `pdf(x)` | yes | no | probability density |
| `log_pdf(x)` | yes | no | log density |
| `pmf(x)` | no | yes | probability mass |
| `log_pmf(x)` | no | yes | log mass |
| `cdf(x)` | yes | yes | `P(X <= x)` |
| `sf(x)` | yes | yes | survival, `P(X > x)`, accurate in the upper tail |
| `ppf(q)` | yes | yes | inverse cdf, `q` in `[0, 1]` |
| `isf(q)` | yes | yes | inverse survival, `ppf(1 - q)` |
| `log_cdf(x)` | yes | yes | log cdf |
| `log_sf(x)` | yes | yes | log survival |
| `mean()` | yes | yes | `E[X]` |
| `variance()` / `std()` | yes | yes | variance and its square root |
| `median()` | yes | yes | `ppf(0.5)`, or a closed form when available |
| `entropy()` | yes | yes | differential / Shannon entropy, in nats |
| `sample(seed=None)` | yes | yes | one variate per row |
| `samples(size, seed=None)` | yes | yes | a width-`size` `Array` per row |

Where a more accurate closed form is available (a native `sf`, `ln_pdf`, or a stable `log_sf`), a distribution binds
it; otherwise the composing defaults apply (`sf = 1 - cdf`, `log_pdf = pdf().log()`, `median = ppf(0.5)`).

Every method is a `pl.Expr`, so it works the same whether the parameters are scalars or per-row columns. See:

* [Column-valued parameters](../user-guide/column-parameters.md): per-row parameterisation.
* [Sampling](../user-guide/sampling.md): the `sample` / `samples` contract and seeding.
* [Nulls and errors](../user-guide/nulls-and-errors.md): null propagation and the invalid-parameter contract.
