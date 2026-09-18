"""Entrypoint for the ``polars_stats`` vs ``scipy.stats`` benchmarks.

```terminal
uv run --group tools -m tools.benchmarks.run                               # all distributions and methods
uv run --group tools -m tools.benchmarks.run normal binomial               # a subset of distributions
uv run --group tools -m tools.benchmarks.run --methods sample density ppf  # a subset of methods
uv run --group tools -m tools.benchmarks.run --regimes scalar column       # a subset of regimes
uv run --group tools -m tools.benchmarks.run normal --rows 1_000_000 10_000_000 --n-samples 5 10 20
uv run --group tools -m tools.benchmarks.run --memory                      # also measure peak RSS
uv run --group tools -m tools.benchmarks.run --format markdown             # write tools/benchmarks/results/<dist>.md
```

The harness does the work; each distribution is one `Comparison` in the registry below.
"""

# pyright: reportUnknownParameterType=false
# See the same suppression in `_harness.py`: every factory below returns a `ScipyFrozen`, which
# aliases scipy-stubs' generic frozen types and carries their partly-Unknown arguments here.
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import numpy as np
from cyclopts import App, Parameter
from scipy.stats import bernoulli, beta, binom, cauchy, expon, geom, lognorm, norm, randint, uniform

from polars_stats import (
    Bernoulli,
    Beta,
    Binomial,
    Cauchy,
    DiscreteUniform,
    Exponential,
    Geometric,
    LogNormal,
    Normal,
    Uniform,
)
from tools.benchmarks._harness import (
    Comparison,
    OutputFormat,
    ParamSpec,
    Sweep,
    measure_cases,
    report,
    require_release_build,
)

if TYPE_CHECKING:
    from tools.benchmarks._harness import Distribution, Params, Result, ScipyFrozen

_ReportFormat = Annotated[OutputFormat, Parameter(name="--format")]

_DEFAULT_SWEEP = Sweep()
"""Frozen, so it is safe as a default: `Sweep` is also the CLI's option surface."""


# Each factory spells its parameters once per side, so it stays regime-agnostic. Module-level, not
# lambdas: a `Comparison` is pickled into the memory subprocess.


def _normal(params: Params) -> tuple[Distribution, ScipyFrozen]:
    return Normal(mu=params.plugin("mu"), sigma=params.plugin("sigma")), norm(
        loc=params.scipy("mu"), scale=params.scipy("sigma")
    )


def _lognormal(params: Params) -> tuple[Distribution, ScipyFrozen]:
    return (
        LogNormal(mu=params.plugin("mu"), sigma=params.plugin("sigma")),
        lognorm(s=params.scipy("sigma"), scale=np.exp(params.scipy("mu"))),
    )


def _cauchy(params: Params) -> tuple[Distribution, ScipyFrozen]:
    return Cauchy(loc=params.plugin("loc"), scale=params.plugin("scale")), cauchy(
        loc=params.scipy("loc"), scale=params.scipy("scale")
    )


def _uniform(params: Params) -> tuple[Distribution, ScipyFrozen]:
    low, high = params.scipy("min"), params.scipy("max")
    return Uniform(min=params.plugin("min"), max=params.plugin("max")), uniform(loc=low, scale=high - low)


def _exponential(params: Params) -> tuple[Distribution, ScipyFrozen]:
    return Exponential(rate=params.plugin("rate")), expon(scale=1.0 / params.scipy("rate"))


def _beta(params: Params) -> tuple[Distribution, ScipyFrozen]:
    return Beta(a=params.plugin("a"), b=params.plugin("b")), beta(a=params.scipy("a"), b=params.scipy("b"))


def _bernoulli(params: Params) -> tuple[Distribution, ScipyFrozen]:
    return Bernoulli(p=params.plugin("p")), bernoulli(params.scipy("p"))


def _binomial(params: Params) -> tuple[Distribution, ScipyFrozen]:
    return Binomial(n=params.plugin_int("n"), p=params.plugin("p")), binom(params.scipy("n"), params.scipy("p"))


def _discrete_uniform(params: Params) -> tuple[Distribution, ScipyFrozen]:
    # scipy's `randint` is half-open on the right where `DiscreteUniform` is inclusive.
    return (
        DiscreteUniform(min=params.plugin_int("min"), max=params.plugin_int("max")),
        randint(low=params.scipy("min"), high=params.scipy("max") + 1),
    )


def _geometric(params: Params) -> tuple[Distribution, ScipyFrozen]:
    return Geometric(p=params.plugin("p")), geom(params.scipy("p"))


# Ordered domains (`min` below `max`) are kept non-overlapping so every draw is a valid
# parameterisation. Keyed by each comparison's own name, which is also its report file name.
REGISTRY: dict[str, Comparison] = {
    comparison.name: comparison
    for comparison in (
        Comparison(
            name="normal",
            params={"mu": ParamSpec(0.0, -1.0, 1.0), "sigma": ParamSpec(1.0, 0.5, 2.0)},
            build=_normal,
        ),
        Comparison(
            name="lognormal",
            params={"mu": ParamSpec(0.0, -1.0, 1.0), "sigma": ParamSpec(1.0, 0.5, 2.0)},
            build=_lognormal,
        ),
        Comparison(
            name="cauchy",
            params={"loc": ParamSpec(0.0, -1.0, 1.0), "scale": ParamSpec(1.0, 0.5, 2.0)},
            build=_cauchy,
            undefined_moments=frozenset({"mean", "variance", "std"}),
        ),
        Comparison(
            name="uniform",
            params={"min": ParamSpec(0.0, -1.0, 0.0), "max": ParamSpec(1.0, 0.5, 1.5)},
            build=_uniform,
        ),
        Comparison(name="exponential", params={"rate": ParamSpec(1.0, 0.5, 2.0)}, build=_exponential),
        Comparison(
            name="beta",
            params={"a": ParamSpec(2.0, 1.0, 4.0), "b": ParamSpec(3.0, 1.0, 4.0)},
            build=_beta,
        ),
        Comparison(name="bernoulli", params={"p": ParamSpec(0.3, 0.1, 0.9)}, build=_bernoulli),
        Comparison(
            name="binomial",
            params={"n": ParamSpec(10, 5, 20, integer=True), "p": ParamSpec(0.3, 0.1, 0.9)},
            build=_binomial,
        ),
        Comparison(
            name="discrete_uniform",
            params={"min": ParamSpec(-2, -5, -1, integer=True), "max": ParamSpec(9, 5, 12, integer=True)},
            build=_discrete_uniform,
        ),
        Comparison(name="geometric", params={"p": ParamSpec(0.3, 0.1, 0.9)}, build=_geometric),
    )
}

app = App(name="bench", help="Benchmark polars_stats against scipy.stats.")


@app.default
def main(
    distributions: list[str] | None = None,
    *,
    sweep: Annotated[Sweep, Parameter(name="*")] = _DEFAULT_SWEEP,
    memory: bool = False,
    output_format: _ReportFormat = "rich",
    output_dir: Path | None = None,
) -> None:
    """Compare each requested distribution's methods against scipy, in each requested regime.

    Cells from different regimes are different code paths on both sides and must never be compared
    against each other. `density` / `log_density` resolve to `pdf` / `pmf` and `log_pdf` / `log_pmf`
    per distribution family.

    Arguments:
        distributions: Distributions to compare (e.g. `normal binomial`). Defaults to all of them.
        sweep: The grid to benchmark. Its fields are the `--rows` / `--n-samples` / `--regimes` /
            `--methods` / `--seed` and budget flags; see `Sweep` and `Budget` for the defaults.
        memory: Also measure peak RSS per contender per cell. Off by default: it spawns one
            subprocess per contender per cell, which dominates the runtime of a full sweep.
        output_format: `rich` prints a coloured table to the terminal; `markdown` / `json` write a
            file per distribution to the output directory.
        output_dir: Where the `markdown` / `json` files are written. Defaults to `tools/benchmarks/results/`.
    """
    require_release_build()
    names = distributions or list(REGISTRY)
    if unknown := [n for n in names if n not in REGISTRY]:
        msg = f"unknown distribution(s): {', '.join(unknown)}. Available: {', '.join(REGISTRY)}"
        raise ValueError(msg)
    results_dir = output_dir or (Path(__file__).parent / "results")

    for name in names:
        comparison = REGISTRY[name]
        results: list[Result] = []
        try:
            for result in measure_cases(comparison, sweep, memory=memory):
                results.append(result)  # noqa: PERF402 - an interrupt must leave the partial list intact
        finally:
            # `finally`, so an interrupt still reports the cells already measured before it stops.
            if results:
                report(comparison, results, sweep, output_format=output_format, output_dir=results_dir)


if __name__ == "__main__":
    app()
