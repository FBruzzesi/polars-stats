"""Entrypoint for the ``polars_stats`` vs ``scipy.stats`` benchmarks.

Compare the sampling methods (`sample` / `samples` vs ``scipy.rvs``) and the value-keyed methods
(`pdf` / `pmf`, `log_pdf` / `log_pmf`, `cdf`, `log_cdf`, `sf`, `log_sf`, `ppf` vs their frozen-scipy
counterparts) on speed and peak memory, sweeping over one or more row counts and sample widths in a
single report::

    uv run --group benchmarks benchmarks/run.py                                   # all distributions and methods
    uv run --group benchmarks benchmarks/run.py normal binomial                   # a subset of distributions
    uv run --group benchmarks benchmarks/run.py --methods sample density ppf      # a subset of methods
    uv run --group benchmarks benchmarks/run.py normal --rows 1_000_000 10_000_000 --n-samples 5 10 20
    uv run --group benchmarks benchmarks/run.py --format markdown                 # write benchmarks/results/<dist>.md

The harness does the work; each distribution is just one `Comparison` in the registry below.
"""

from __future__ import annotations

from math import exp
from pathlib import Path
from typing import Annotated, get_args

from cyclopts import App, Parameter
from scipy.stats import bernoulli, beta, binom, expon, geom, lognorm, norm, uniform

from benchmarks._harness import ALL_METHODS, Comparison, Method, OutputFormat, Sweep, emit, run_comparison
from polars_stats import Bernoulli, Beta, Binomial, Exponential, Geometric, LogNormal, Normal, Uniform

# `--rows 1 2 3` (multiple tokens per flag), not `--rows 1 --rows 2`; without this a list option takes
# a single token and the rest leak onto the positional `distributions`.
_MultiInt = Annotated[list[int], Parameter(consume_multiple=True)]
_MultiMethod = Annotated[list[Method], Parameter(consume_multiple=True)]

# One row per distribution: the polars_stats instance and the matching frozen scipy distribution,
# reparameterised to scipy's convention. Adding a distribution is one entry here.
REGISTRY: dict[str, Comparison] = {
    "normal": Comparison("normal", Normal(mu=0.0, sigma=1.0), norm(loc=0.0, scale=1.0)),
    "lognormal": Comparison("lognormal", LogNormal(mu=0.0, sigma=1.0), lognorm(s=1.0, scale=exp(0.0))),
    "uniform": Comparison("uniform", Uniform(min=0.0, max=1.0), uniform(loc=0.0, scale=1.0)),
    "exponential": Comparison("exponential", Exponential(rate=1.0), expon(scale=1.0)),
    "beta": Comparison("beta", Beta(a=2.0, b=3.0), beta(a=2.0, b=3.0)),
    "bernoulli": Comparison("bernoulli", Bernoulli(p=0.3), bernoulli(0.3)),
    "binomial": Comparison("binomial", Binomial(n=10, p=0.3), binom(10, 0.3)),
    "geometric": Comparison("geometric", Geometric(p=0.3), geom(0.3)),
}

app = App(name="bench", help="Benchmark polars_stats sampling against scipy.stats.")


@app.default
def main(  # noqa: PLR0913
    distributions: list[str] | None = None,
    *,
    methods: _MultiMethod | None = None,
    rows: _MultiInt | None = None,
    n_samples: _MultiInt | None = None,
    iterations: int = 50,
    seed: int = 0,
    format: OutputFormat = "rich",  # noqa: A002
    output_dir: Path | None = None,
) -> None:
    """Compare each requested distribution's methods against scipy.

    Arguments:
        distributions: Distributions to compare (e.g. `normal binomial`). Defaults to all of them.
        methods: Methods to compare (e.g. `--methods sample density log_sf ppf`); `density` /
            `log_density` resolve to `pdf` / `pmf` and `log_pdf` / `log_pmf` per distribution
            family. Defaults to all of them.
        rows: Row counts to sweep over (e.g. `--rows 1_000_000 10_000_000`). Defaults to `[1_000_000]`.
        n_samples: Sample widths to sweep over for `samples` (e.g. `--n-samples 5 10 20`). Defaults to `[10]`.
        iterations: Timed runs per cell; runtime is reported as p50 +/- std.
        seed: Seed for the samplers and the value-keyed evaluation inputs (reproducible).
        format: `rich` prints a coloured table to the terminal; `markdown` / `json` write a file per
            distribution to the output directory.
        output_dir: Where the `markdown` / `json` files are written. Defaults to `benchmarks/results/`.
    """
    names = distributions or list(REGISTRY)
    if unknown := [n for n in names if n not in REGISTRY]:
        msg = f"unknown distribution(s): {', '.join(unknown)}. Available: {', '.join(REGISTRY)}"
        raise ValueError(msg)
    selected = tuple(methods) if methods else ALL_METHODS
    if bad := [m for m in selected if m not in get_args(Method)]:
        msg = f"unknown method(s): {', '.join(bad)}. Available: {', '.join(get_args(Method))}"
        raise ValueError(msg)

    sweep = Sweep(
        rows=tuple(rows) if rows else (1_000_000,),
        n_samples=tuple(n_samples) if n_samples else (10,),
        iterations=iterations,
        seed=seed,
    )
    results_dir = output_dir or (Path(__file__).parent / "results")

    for name in names:
        results = run_comparison(REGISTRY[name], sweep, methods=selected)
        emit(REGISTRY[name], results, sweep, fmt=format, output_dir=results_dir)


if __name__ == "__main__":
    app()
