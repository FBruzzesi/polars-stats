---
icon: lucide/list-checks
---

# Parameters and contracts

What every distribution accepts, what it returns, and what happens when an input is missing or invalid. This page is
the lookup table; the how-to guides show what to do about it.

## Accepted inputs

Constructor parameters and value-keyed method arguments follow the same coercion rules, with one difference: a float
parameter rejects an `int`, while a method argument accepts either.

| Input | Float parameter (`mu`, `sigma`, `p`, `rate`, `a`, `b`, `min`, `max`) | Count parameter (`n`) | Method argument (`x`, `q`) |
|---|---|---|---|
| `float` | accepted | `TypeError` | accepted |
| `int` | `TypeError` | accepted | accepted |
| `bool` | `TypeError` | `TypeError` | `TypeError` |
| `str` | read as `pl.col(name)` | read as `pl.col(name)` | read as `pl.col(name)` |
| `pl.Expr` | passed through | passed through | passed through |
| `pl.Series` | wrapped as `pl.lit(series)` | wrapped as `pl.lit(series)` | wrapped as `pl.lit(series)` |
| anything else | `TypeError` | `TypeError` | `TypeError` |

A `str` is always a column reference, never a literal value. A Python scalar is expanded to a row-aligned expression
rather than a broadcast literal, which is what keeps every method elementwise under `over` and `group_by`
(see [Architecture](../explanation/architecture.md#column-valued-parameters)).

Numeric columns of any width are cast to `Float64` (or `Int64` for `n`) at evaluation, so an integer column works
wherever a float is expected. A non-numeric column raises at evaluation: Polars fails the query with
`InvalidOperationError` rather than returning nulls.

## Parameter validity

Values are validated at *evaluation*, not at construction, and identically for scalar and column-valued parameters.

| Distribution | Required | Also required |
|---|---|---|
| `Beta(a, b)` | `a > 0`, `b > 0` | both finite |
| `Exponential(rate)` | `rate > 0` | finite |
| `LogNormal(mu, sigma)` | `sigma > 0` | both finite |
| `Normal(mu, sigma)` | `sigma > 0` | both finite |
| `Uniform(min, max)` | `max > min` | `max - min` finite |
| `Bernoulli(p)` | `0 <= p <= 1` | |
| `Binomial(n, p)` | `n >= 0`, `0 <= p <= 1` | `n` integral |

A violation raises `ComputeError` and fails the whole evaluation. See
[nulls, NaNs and errors](#nulls-nans-and-errors) below.

## Sampling

`sample(seed)` returns one variate per row; `samples(size, seed)` returns `Array(inner=<element dtype>, shape=size)`.
A `size <= 0` raises `ValueError` at call time.

Element dtype is per distribution and is not normalised to `Float64`:

| Distribution | Sample dtype |
|---|---|
| `Bernoulli` | `Boolean` |
| `Binomial` | `UInt64` |
| `Beta`, `Exponential`, `LogNormal`, `Normal`, `Uniform` | `Float64` |

| Aspect | Behaviour |
|---|---|
| Output length | the surrounding context: frame length under `select` / `with_columns`, partition length under `over` / `group_by` |
| `seed=<int>` | deterministic across OS, architecture, chunking, thread count, and engine (in-memory or streaming) |
| `seed=None` | non-reproducible by design (OS entropy), resolved once per call |
| Output name | `"sample"` / `"samples"` when every parameter is a scalar, otherwise the first parameter expression's root name |
| `samples(1, seed)` | bit-identical to `sample(seed)`; growing `size` extends each row's array without changing existing draws |
| Null parameter on a row | `sample` yields `null`; `samples` yields a `null` array, not an array of nulls |

## Nulls, NaNs and errors

**`null` is reserved for missing inputs; an invalid parameter raises.** A silent null from a bad parameter would be
indistinguishable from a legitimately missing input, and would propagate wrong answers downstream.

| Situation | When detected | Behaviour |
|---|---|---|
| Wrong parameter *type* (a `list`, an `int` for a float parameter, a `bool`) | Python `__init__` | `TypeError`, no query runs |
| Invalid parameter *value*, scalar or one column row | Rust evaluation | `ComputeError`, fails the whole evaluation, never silently nulls |
| `null` value or quantile argument on a row | per row | `null` on that row |
| `null` parameter on a row | per row | `null` on that row, no error |
| `NaN` value or quantile argument on a row | per row | `NaN` on that row (matches scipy) |
| Non-numeric column as an argument or parameter | evaluation | Polars raises `InvalidOperationError` |
| `q` outside `[0, 1]` in `ppf` / `isf` | per row | `null`, guaranteed for every distribution and both parameter regimes (pinned by `tests/property/ppf_domain_test.py`). `q` exactly `0` or `1` is in range and maps to a support bound |
| `x` outside the support (e.g. `pdf` below a `Uniform`'s `min`) | per row | `0.0` (matches scipy) |
| `pmf(3.5)` for a discrete distribution | per row | `0.0` (matches scipy) |
| Deep-tail underflow (`sf` of a 60-sigma event) | per row | `0.0`; use `log_sf` to keep resolution |

Every distribution shipped today has finite moments on its valid parameter range, so this contract is exhaustive for
them. The policy for distributions whose moments can be undefined is in
[Design notes](../explanation/design.md#moments-that-are-undefined).

## Related

* [Handle nulls and errors](../how-to/nulls-and-errors.md): what to do about each of these cases.
* [Design notes](../explanation/design.md#invalid-parameters-raise-they-never-silently-null): why raising, not nulling.
