---
icon: lucide/book-open
---

# API reference

Technical description of the machinery. The generated docstrings live on two pages, [Continuous](continuous.md) and
[Discrete](discrete.md); the input types, dtypes, and error contracts are in
[Parameters and contracts](parameters-and-contracts.md).

For worked examples, see the [tutorial](../tutorial.md) and the [How-to guides](../how-to/index.md).

## Catalogue

| Distribution | Kind | Parameters | scipy equivalent |
|---|---|---|---|
| `Beta(a, b)` | continuous | `a > 0`, `b > 0` | `beta(a, b)` |
| `Exponential(rate)` | continuous | `rate > 0` | `expon(scale=1 / rate)` |
| `LogNormal(mu, sigma)` | continuous | `sigma > 0` | `lognorm(s=sigma, scale=exp(mu))` |
| `Normal(mu, sigma)` | continuous | `sigma > 0` | `norm(loc=mu, scale=sigma)` |
| `Uniform(min, max)` | continuous | `max > min` | `uniform(loc=min, scale=max - min)` |
| `Bernoulli(p)` | discrete | `0 <= p <= 1` | `bernoulli(p)` |
| `Binomial(n, p)` | discrete | `n >= 0`, `0 <= p <= 1` | `binom(n, p)` |
| `DiscreteUniform(min, max)` | discrete | `min <= max`, both inclusive | `randint(low=min, high=max + 1)`; **`max` is inclusive** |
| `Geometric(p)` | discrete | `0 < p <= 1` | `geom(p)` |

Each class names its parameters after the distribution's conventional parameters. `Normal` and `LogNormal` default to
`mu=0.0, sigma=1.0`; the others have no defaults. Parameter values are validated at evaluation, not at construction.

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

Argument-free statistics return one value per row of parameters: with column-valued parameters, `mean()` yields the
mean of a different distribution on every row.

## Compatibility

| Dimension | Values |
|---|---|
| Python | 3.10 to 3.14 (per `requires-python`), single abi3 wheel |
| Polars | `>=1.15` (the `pyo3-polars` ABI floor) |
| OS | wheels for Linux x86_64/aarch64, macOS arm64/x86_64, Windows x86_64 |
| Runtime dependencies | `polars` only |

!!! warning "Known limitation on polars >= 1.44"

    On polars 1.44.0 and newer, `Geometric` may **return a value instead of raising
    `ComputeError`** when a *column-valued* parameter is invalid on a row the selected branch does
    not cover.

    Polars 1.44.0 ([pola-rs/polars#28498](https://github.com/pola-rs/polars/pull/28498)) masks the arms
    of a `when/then/otherwise` to null on the rows an arm does not select, so a validator reached only
    from inside an arm never sees the offending row. `Geometric` assembles its value-keyed closed
    forms (`pmf`, `log_pmf`, `cdf`, `log_cdf`, `sf`, `log_sf`, `ppf`, `isf`) as branching Polars
    expressions, so it is affected.

    **Not affected:** every other distribution (they compute in Rust and validate unconditionally),
    `Geometric`'s own moments (`mean`, `variance`, `std`, `median`, `entropy`), and every *valid*
    computation, whose results are unchanged. `Bernoulli`, `Exponential` and `Uniform` were affected
    up to and including v0.0.2 and no longer are: their closed forms now compute in Rust.

    One narrower case survives for *every* distribution and *both* parameter spellings, scalar
    included: a **null or `NaN` evaluation point** is masked out of the plugin's input by the wrapper
    that gives you `null -> null` and `NaN -> NaN`, so an invalid parameter goes unreported on those
    rows. With a column parameter that is per row; with a scalar one it takes the whole batch, so
    `Bernoulli(p=1.5).pmf(col)` returns `[nan, nan]` over an all-`NaN` column but still raises as soon
    as one evaluation point is finite.

    **Workarounds:** pin `polars<1.44` yourself, or validate column parameters before passing them.

    Tracked as [pola-rs/polars#29005](https://github.com/pola-rs/polars/issues/29005). The limitation
    goes away once `Geometric` moves its closed forms into Rust, which is in progress.
