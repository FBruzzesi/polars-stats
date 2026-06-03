# polars-stats

`scipy.stats`-style probability distributions as native [Polars](https://pola.rs) expressions, with **column-valued
parameters**: any distribution parameter can be a scalar or a Polars expression, so a single instance describes a
different distribution per row, fully lazy and vectorised.

> **Alpha.** The public API may change before `1.0`.

## Install

```bash
pip install polars-stats
```

Requires `polars >= 1.15` and Python `>= 3.10`.

## Example

```python
import polars as pl
from polars_stats import Normal

df = pl.DataFrame({"mu": [0.0, 10.0], "sigma": [1.0, 2.0], "x": [0.5, 11.0]})

norm = Normal(mean="mu", std_dev="sigma")

df.with_columns(
    density=norm.pdf("x"),
    tail_prob=norm.sf("x"),
)
```

## Documentation

Full docs at [fbruzzesi.github.io/polars-stats](https://fbruzzesi.github.io/polars-stats/): the
[distribution catalogue and method surface](https://fbruzzesi.github.io/polars-stats/distributions/), the
[API reference](https://fbruzzesi.github.io/polars-stats/reference/), and the
[architecture](https://fbruzzesi.github.io/polars-stats/architecture/) and
[design notes](https://fbruzzesi.github.io/polars-stats/design/).

## License

This project is licensed under the MIT license.
