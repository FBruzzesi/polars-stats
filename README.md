# polars-stats

`scipy.stats`-style probability distributions as native [Polars](https://pola.rs) expressions, with **column-valued
parameters**: any distribution parameter can be a scalar or a Polars expression, so a single instance describes a
different distribution per row, fully lazy and vectorised.

The math runs in Rust on top of [`statrs`](https://docs.rs/statrs); the Python layer is a thin, typed surface of
distribution classes.

> **Alpha.** The public API may change before `1.0`.

## Install

```bash
pip install polars-stats
```

Requires `polars >= 1.15` and Python `>= 3.10`. Wheels ship for Linux, macOS, and Windows; to build from source see the
[contributing guide](https://fbruzzesi.github.io/polars-stats/contributing/).

## Example

Per-row parameters: each row is scored against its own `Normal`, in one vectorised pass without leaving the lazy engine.

```python
import polars as pl
from polars_stats import Normal

df = pl.DataFrame({"mu": [0.0, 10.0], "sigma": [1.0, 2.0], "x": [0.5, 11.0]})

norm = Normal(mean="mu", std_dev="sigma")

df.with_columns(
    density=norm.pdf("x"),
    tail_prob=norm.sf(pl.col("x") * 2),
)
```

A method argument and a parameter both accept a scalar, a column name (`str`), or a `pl.Expr`, and every method returns
a `pl.Expr`, so it composes inside any `LazyFrame` query.

## Documentation

Full docs at [fbruzzesi.github.io/polars-stats](https://fbruzzesi.github.io/polars-stats/): the
[distribution catalogue and method surface](https://fbruzzesi.github.io/polars-stats/distributions/), the
[API reference](https://fbruzzesi.github.io/polars-stats/reference/), and the
[architecture](https://fbruzzesi.github.io/polars-stats/architecture/) and
[design notes](https://fbruzzesi.github.io/polars-stats/design/).

## License

This project is licensed under the MIT license.
