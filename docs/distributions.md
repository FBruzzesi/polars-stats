---
icon: lucide/chart-spline
---

# Distributions

## Catalogue

Shipped today:

| Distribution | Kind | scipy equivalent |
|---|---|---|
| `Normal(mean, std_dev)` | continuous | `norm(loc=mean, scale=std_dev)` |
| `LogNormal(mu, sigma)` | continuous | `lognorm(s=sigma, scale=exp(mu))` |
| `Uniform(min, max)` | continuous | `uniform(loc=min, scale=max - min)` |
| `Bernoulli(p)` | discrete | `bernoulli(p)` |

Each class names its parameters after the distribution's conventional parameters; the `scipy equivalent` column gives
the `scipy.stats` translation, and each class docstring states it too.

## Method surface

Every distribution exposes the same surface, defined on the base classes. Value-keyed methods take a scalar, a column
name (`str`), or a `pl.Expr`; argument-free statistics take none. All return a `pl.Expr`.

| Method | Continuous | Discrete | Meaning |
|---|---|---|---|
| `pdf(x)` | yes | no | probability density |
| `pmf(x)` | no | yes | probability mass |
| `cdf(x)` | yes | yes | `P(X <= x)` |
| `sf(x)` | yes | yes | survival, `P(X > x)`, accurate in the upper tail |
| `ppf(q)` | yes | yes | inverse cdf, `q` in `[0, 1]` |
| `isf(q)` | yes | yes | inverse survival, `ppf(1 - q)` |
| `log_pdf(x)` | yes | no | native log density (avoids tail underflow) |
| `log_pmf(x)` | no | yes | native log mass |
| `log_cdf(x)` | yes | yes | log cdf |
| `log_sf(x)` | yes | yes | log survival |
| `mean()` | yes | yes | `E[X]` |
| `variance()` / `std()` | yes | yes | variance and its square root |
| `median()` | yes | yes | `ppf(0.5)`, or a closed form when available |
| `entropy()` | yes | yes | differential / Shannon entropy, in nats |
| `sample(seed=None)` | yes | yes | one variate per row |
| `samples(size, seed=None)` | yes | yes | a width-`size` `Array` per row |

Where a more accurate closed form is available (a native `sf`, `ln_pdf`, or `median`), a distribution binds it;
otherwise the composing defaults apply (`sf = 1 - cdf`, `log_pdf = pdf().log()`, `median = ppf(0.5)`).

Every method is a `pl.Expr`, so it works the same whether the parameters are scalars or per-row columns. See:

* [Column-valued parameters](user-guide/column-parameters.md): per-row parameterisation.
* [Sampling](user-guide/sampling.md): the `sample` / `samples` contract and seeding.
* [Nulls and errors](user-guide/nulls-and-errors.md): null propagation and the invalid-parameter contract.
