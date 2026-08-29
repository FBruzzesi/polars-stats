---
icon: lucide/arrow-right-left
---

# Migrate from scipy.stats

Port `scipy.stats` code that runs against Polars data. The method names line up closely; the parameterisation is where
the traps are.

The `scipy` snippets on this page are shown for comparison and are not executed when the docs are built; the
`polars-stats` ones are.

## Translate the method names

| `scipy.stats` | `polars-stats` | Note |
|---|---|---|
| `pdf(x)` / `pmf(k)` | `pdf(x)` / `pmf(k)` | same |
| `logpdf` / `logpmf` | `log_pdf` / `log_pmf` | underscore |
| `cdf` / `logcdf` | `cdf` / `log_cdf` | underscore |
| `sf` / `logsf` | `sf` / `log_sf` | underscore |
| `ppf` / `isf` | `ppf` / `isf` | same |
| `mean()` / `median()` / `std()` | `mean()` / `median()` / `std()` | same |
| `var()` | `variance()` | renamed |
| `entropy()` | `entropy()` | same, in nats |
| `rvs(size=n, random_state=s)` | `sample(seed=s)` / `samples(n, seed=s)` | see [Sample reproducibly](sampling.md) |
| `fit`, `interval`, `moment`, `stats`, `expect`, `support` | none | see [what is missing](#what-has-no-equivalent) |

## Translate the parameters

`scipy` parameterises through `loc` / `scale` shims; `polars-stats` uses each distribution's conventional parameters.
Some of these translations are not identities, so check this table rather than guessing:

| `scipy.stats` | `polars-stats` | Watch out |
|---|---|---|
| `norm(loc=mu, scale=sigma)` | `Normal(mu=mu, sigma=sigma)` | same meaning |
| `lognorm(s=sigma, scale=exp(mu))` | `LogNormal(mu=mu, sigma=sigma)` | `scale` is `exp(mu)`, not `mu` |
| `expon(scale=1 / rate)` | `Exponential(rate=rate)` | **inverted**: `scale` is `1 / rate` |
| `uniform(loc=min, scale=max - min)` | `Uniform(min=min, max=max)` | **`scale` is the width**, not the upper bound |
| `beta(a, b)` | `Beta(a=a, b=b)` | same meaning |
| `bernoulli(p)` | `Bernoulli(p=p)` | same meaning |
| `binom(n, p)` | `Binomial(n=n, p=p)` | same meaning |
| `geom(p)` | `Geometric(p=p)` | same meaning; `p = 0` raises here, `scipy` allows it |
| `randint(low=min, high=max + 1)` | `DiscreteUniform(min=min, max=max)` | **`high` is exclusive, `max` is inclusive**: pass `max = high - 1` |

Two further differences apply everywhere:

* **No `loc` / `scale` shifting.** `scipy` lets you shift and scale any distribution (`expon(loc=5, scale=2)`).
    `polars-stats` does not, so shift the values yourself: `Exponential(rate=0.5).cdf(pl.col("x") - 5.0)`.
* **Floats are strict.** `scipy.stats.norm(0, 1)` is fine; `ps.Normal(0, 1)` raises `TypeError`. Write `0.0, 1.0`. A
    count parameter is the opposite: `Binomial(n=10, p=0.3)` needs an `int` for `n`.

## Port a whole-column evaluation

Before, leaving Polars and coming back:

```python
import numpy as np
from scipy import stats

x = df["x"].to_numpy()
df = df.with_columns(upper_tail=pl.Series(stats.norm(loc=0.0, scale=1.0).sf(x)))
```

After, staying inside the expression engine:

```python exec="yes" source="above" session="migrate" result="python"
import polars as pl
import polars_stats as ps

df = pl.DataFrame({"x": [-1.0, 0.5, 2.0]})

print(df.with_columns(upper_tail=ps.Normal(mu=0.0, sigma=1.0).sf("x")))
```

## Port per-row parameters

`scipy` broadcasts parameter arrays, so this part is not about vectorisation. It is about the round trip: three
`to_numpy()` calls, a `pl.Series` on the way back, and manual realignment if any filter comes between.

```python
x = df["x"].to_numpy()
mu = df["mu"].to_numpy()
sigma = df["sigma"].to_numpy()
df = df.with_columns(density=pl.Series(stats.norm(loc=mu, scale=sigma).pdf(x)))
```

After, as one expression that a `LazyFrame` can plan:

```python exec="yes" source="above" session="migrate" result="python"
per_row = pl.DataFrame({"mu": [0.0, 10.0], "sigma": [1.0, 2.0], "x": [0.5, 11.2]})

print(per_row.with_columns(density=ps.Normal(mu="mu", sigma="sigma").pdf("x")))
```

## Port a per-group fit

`scipy` has `fit`; `polars-stats` does not. For the distributions shipped today the estimators are Polars expressions,
which keeps the fit and the scoring in one query:

```python
# scipy: group, fit, score, reassemble
scored = []
for name, part in df.group_by("sensor"):
    mu, sigma = stats.norm.fit(part["reading"].to_numpy())
    scored.append(
        part.with_columns(tail=pl.Series(stats.norm(mu, sigma).sf(part["reading"])))
    )
df = pl.concat(scored)
```

```python exec="yes" source="above" session="migrate" result="python"
history = pl.DataFrame(
    {
        "sensor": ["a", "a", "a", "b", "b", "b"],
        "reading": [1.0, 1.2, 0.9, 10.0, 10.5, 9.5],
    }
)

print(
    history.with_columns(
        mu=pl.col("reading").mean().over("sensor"),
        sigma=pl.col("reading").std(ddof=0).over("sensor"),
    ).with_columns(tail=ps.Normal(mu="mu", sigma="sigma").sf("reading"))
)
```

`scipy.stats.norm.fit` returns the maximum-likelihood estimate, which divides by `n`, so pass `ddof=0` to match it.
Polars defaults to `ddof=1`.

## Behaviour differences to expect

| | `scipy.stats` | `polars-stats` |
|---|---|---|
| Missing data | no null type; `nan` in, `nan` out | `null` in, `null` out, and `NaN` in, `NaN` out |
| Invalid parameter (`scale=-1`) | returns `nan` silently | raises `ComputeError` and fails the query |
| Return type | NumPy array | `pl.Expr` |
| Randomness | `random_state` / global NumPy state | per-call `seed` only, no global state |
| Accuracy | reference implementation | matched to `1e-12` absolute in the parity suite, relaxed to `1e-9` / `1e-6` for erf-based and search-based `ppf` methods |
| `DiscreteUniform.median()` | `randint.median()` is `ppf(0.5)`, a support point | the midpoint `(min + max) / 2`, which for an even support size is not a support point |
| Discrete `ppf(0)` / `isf(1)` | the below-support sentinel `low - 1` | clamped to the support, so `ppf(0)` is `min` and `isf(1)` is `min` |

## What has no equivalent

* `fit` and other estimators: compute them as Polars expressions, as above.
* `interval`, `moment(n)`, `stats(moments=...)`, `expect(...)`, `support()`: not implemented.
* `loc` / `scale` shifting of arbitrary distributions: shift the input instead.
* Distributions outside the [catalogue](../reference/index.md#catalogue), which is short but growing.
* Multivariate distributions, `rv_histogram`, and hypothesis tests: out of scope.

Where a distribution or method is missing, `scipy` remains the right tool; the two libraries coexist in one project
without conflict.
