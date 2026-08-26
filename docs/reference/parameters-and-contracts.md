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
| `int` | `TypeError` | accepted (`0` to `2**63 - 1`) | accepted |
| `bool` | `TypeError` | `TypeError` | `TypeError` |
| `str` | read as `pl.col(name)` | read as `pl.col(name)` | read as `pl.col(name)` |
| `pl.Expr` | passed through | passed through | passed through |
| `pl.Series` | wrapped as `pl.lit(series)` | wrapped as `pl.lit(series)` | wrapped as `pl.lit(series)` |
| anything else | `TypeError` | `TypeError` | `TypeError` |

A `str` is always a column reference, never a literal value. A Python scalar becomes `pl.lit(value)`, a length-1
scalar column that the Rust plugin broadcasts to the call's row count
(see [Architecture](../explanation/architecture.md#column-valued-parameters)).

Polars' own scalar semantics then apply. An expression whose inputs are *all* constant is a scalar column:
`df.select(Normal(0.0, 1.0).mean())` returns **one row**, `group_by().agg()` returns a scalar rather than a list per
group, and a 0-row frame still returns one row. Any column-valued input sets the length instead.

`sample()` and `samples()` are the exception. They pass a per-row index as a hidden full-length input, so they are
full height whatever the parameters are, and a 0-row frame returns no rows.

A length-1 *expression* is accepted wherever a column is and broadcasts rather than truncating, so
`Normal(mu=pl.col("mu").mean(), sigma=1.0).pdf("x")` is full height. Lengths that are neither equal nor 1 raise.

!!! warning "Length-1 inputs inside `over()` and `group_by().agg()` need polars 1.34"

    Before polars 1.34 a length-1 input is mishandled by polars itself once the expression is evaluated *inside* a
    partition context. Depending on the distribution and method you get either a `PanicException` or, below 1.33,
    values returned in group order rather than scattered back to row order. Both defects are fixed upstream and
    need no change here.

    Nothing else is affected. `select` and `with_columns` broadcast correctly on every supported version, and so
    does the pattern of aggregating first and applying the distribution to the result:

    ```python
    (
        frame.group_by("g")
        .agg(mu=pl.col("x").mean(), sigma=pl.col("x").std())
        .with_columns(p=ps.Normal(mu="mu", sigma="sigma").cdf(1.0))
    )
    ```

    Computing a parameter with `.over("g")` and using it in a plain `select` is fine too; it is only putting the
    distribution expression itself inside `over()` or `agg()` that is affected.

Numeric columns of any width are cast to `Float64` at evaluation, so an integer column works wherever a float is
expected. The count parameter `n` is the exception: its column must already hold integers, of any width up to
`UInt64`, because casting a float one would silently truncate a fractional count. The rule is judged on the
dtype, so a float `n` column raises even when every value in it is null; a `Null`-dtype column propagates nulls
like any other parameter. A non-numeric column raises at
evaluation: Polars fails the query with `InvalidOperationError` rather than returning nulls.

## Parameter validity

Values are validated at *evaluation*, not at construction, and identically for scalar and column-valued parameters.
The count parameter `n` is the exception: a Python `int` outside `[0, 2**63 - 1]` raises `ValueError` at
construction, since it is expanded to a `UInt64` column and passed to the fast paths as a kwarg. An `n` *column*
may hold any count its dtype can, up to `UInt64`.

| Distribution | Required | Also required |
|---|---|---|
| `Beta(a, b)` | `a > 0`, `b > 0` | both finite |
| `Exponential(rate)` | `rate > 0` | finite |
| `LogNormal(mu, sigma)` | `sigma > 0` | both finite |
| `Normal(mu, sigma)` | `sigma > 0` | both finite |
| `Uniform(min, max)` | `max > min` | `max - min` finite |
| `Bernoulli(p)` | `0 <= p <= 1` | |
| `Binomial(n, p)` | `n >= 0`, `0 <= p <= 1` | `n` integral |
| `Geometric(p)` | `0 < p <= 1` | `p = 0` rejected, unlike `Bernoulli` |

A violation raises `ComputeError` and fails the whole evaluation. See
[nulls, NaNs and errors](#nulls-nans-and-errors) below.

## Sampling

`sample(seed)` returns one variate per row; `samples(size, seed)` returns `Array(inner=<element dtype>, shape=size)`.
A `size <= 0` raises `ValueError` at call time.

Element dtype is per distribution and is not normalised to `Float64`:

| Distribution | Sample dtype |
|---|---|
| `Bernoulli` | `Boolean` |
| `Binomial`, `Geometric` | `UInt64` |
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
| Deep-tail underflow (`sf` of a 60-sigma event) | per row | `0.0`; `log_sf` keeps resolution where it has a stable form, see [Numerical accuracy](../explanation/accuracy.md) |

Every distribution shipped today has finite moments on its valid parameter range, so this contract is exhaustive for
them. The policy for distributions whose moments can be undefined is in
[Design notes](../explanation/design.md#moments-that-are-undefined).

## Related

* [Handle nulls and errors](../how-to/nulls-and-errors.md): what to do about each of these cases.
* [Design notes](../explanation/design.md#invalid-parameters-raise-they-never-silently-null): why raising, not nulling.
