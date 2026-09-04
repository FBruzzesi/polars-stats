"""Entrypoint for the ``polars_stats`` vs ``scipy.stats`` benchmarks.

```terminal
uv run --group benchmarks benchmarks/run.py                                   # all distributions and methods
uv run --group benchmarks benchmarks/run.py normal binomial                   # a subset of distributions
uv run --group benchmarks benchmarks/run.py --methods sample density ppf      # a subset of methods
uv run --group benchmarks benchmarks/run.py --regimes scalar column           # a subset of regimes
uv run --group benchmarks benchmarks/run.py normal --rows 1_000_000 10_000_000 --n-samples 5 10 20
uv run --group benchmarks benchmarks/run.py --memory                          # also measure peak RSS
uv run --group benchmarks benchmarks/run.py --format markdown                 # write benchmarks/results/<dist>.md
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
from scipy.stats import bernoulli, beta, binom, expon, geom, lognorm, norm, randint, uniform

from benchmarks._harness import Comparison, OutputFormat, ParamSpec, Sweep, emit, measure_cases, require_release_build
from polars_stats import Bernoulli, Beta, Binomial, DiscreteUniform, Exponential, Geometric, LogNormal, Normal, Uniform

if TYPE_CHECKING:
    from benchmarks._harness import Distribution, Params, Result, ScipyFrozen

_ReportFormat = Annotated[OutputFormat, Parameter(name="--format")]

_DEFAULT_SWEEP = Sweep()
"""Frozen, so it is safe as a default: `Sweep` is also the CLI's option surface."""


# Each factory spells its parameters once per side, so it stays regime-agnostic. Module-level, not
# lambdas: a `Comparison` is pickled into the memory subprocess.


def _normal(p: Params) -> tuple[Distribution, ScipyFrozen]:
    return Normal(mu=p.plugin("mu"), sigma=p.plugin("sigma")), norm(loc=p.scipy("mu"), scale=p.scipy("sigma"))


def _lognormal(p: Params) -> tuple[Distribution, ScipyFrozen]:
    return (
        LogNormal(mu=p.plugin("mu"), sigma=p.plugin("sigma")),
        lognorm(s=p.scipy("sigma"), scale=np.exp(p.scipy("mu"))),
    )


def _uniform(p: Params) -> tuple[Distribution, ScipyFrozen]:
    low, high = p.scipy("min"), p.scipy("max")
    return Uniform(min=p.plugin("min"), max=p.plugin("max")), uniform(loc=low, scale=high - low)


def _exponential(p: Params) -> tuple[Distribution, ScipyFrozen]:
    return Exponential(rate=p.plugin("rate")), expon(scale=1.0 / p.scipy("rate"))


def _beta(p: Params) -> tuple[Distribution, ScipyFrozen]:
    return Beta(a=p.plugin("a"), b=p.plugin("b")), beta(a=p.scipy("a"), b=p.scipy("b"))


def _bernoulli(p: Params) -> tuple[Distribution, ScipyFrozen]:
    return Bernoulli(p=p.plugin("p")), bernoulli(p.scipy("p"))


def _binomial(p: Params) -> tuple[Distribution, ScipyFrozen]:
    return Binomial(n=p.plugin_int("n"), p=p.plugin("p")), binom(p.scipy("n"), p.scipy("p"))


def _discrete_uniform(p: Params) -> tuple[Distribution, ScipyFrozen]:
    # scipy's `randint` is half-open on the right where `DiscreteUniform` is inclusive.
    return (
        DiscreteUniform(min=p.plugin_int("min"), max=p.plugin_int("max")),
        randint(low=p.scipy("min"), high=p.scipy("max") + 1),
    )


def _geometric(p: Params) -> tuple[Distribution, ScipyFrozen]:
    return Geometric(p=p.plugin("p")), geom(p.scipy("p"))


# Ordered domains (`min` below `max`) are kept non-overlapping so every draw is a valid
# parameterisation.
REGISTRY: dict[str, Comparison] = {
    "normal": Comparison(
        name="normal",
        params={"mu": ParamSpec(0.0, -1.0, 1.0), "sigma": ParamSpec(1.0, 0.5, 2.0)},
        build=_normal,
    ),
    "lognormal": Comparison(
        name="lognormal",
        params={"mu": ParamSpec(0.0, -1.0, 1.0), "sigma": ParamSpec(1.0, 0.5, 2.0)},
        build=_lognormal,
    ),
    "uniform": Comparison(
        name="uniform",
        params={"min": ParamSpec(0.0, -1.0, 0.0), "max": ParamSpec(1.0, 0.5, 1.5)},
        build=_uniform,
    ),
    "exponential": Comparison(
        name="exponential",
        params={"rate": ParamSpec(1.0, 0.5, 2.0)},
        build=_exponential,
    ),
    "beta": Comparison(
        name="beta",
        params={"a": ParamSpec(2.0, 1.0, 4.0), "b": ParamSpec(3.0, 1.0, 4.0)},
        build=_beta,
    ),
    "bernoulli": Comparison(
        name="bernoulli",
        params={"p": ParamSpec(0.3, 0.1, 0.9)},
        build=_bernoulli,
    ),
    "binomial": Comparison(
        name="binomial",
        params={"n": ParamSpec(10, 5, 20, integer=True), "p": ParamSpec(0.3, 0.1, 0.9)},
        build=_binomial,
    ),
    "discrete_uniform": Comparison(
        name="discrete_uniform",
        params={"min": ParamSpec(-2, -5, -1, integer=True), "max": ParamSpec(9, 5, 12, integer=True)},
        build=_discrete_uniform,
    ),
    "geometric": Comparison(
        name="geometric",
        params={"p": ParamSpec(0.3, 0.1, 0.9)},
        build=_geometric,
    ),
}

if mismatched := {key: comp.name for key, comp in REGISTRY.items() if key != comp.name}:
    # A key that disagrees with its own name would write to the wrong report file.
    _msg = f"REGISTRY keys must equal their Comparison.name; mismatched: {mismatched}"
    raise RuntimeError(_msg)

app = App(name="bench", help="Benchmark polars_stats against scipy.stats.")


@app.default
def main(
    distributions: list[str] | None = None,
    *,
    sweep: Annotated[Sweep, Parameter(name="*")] = _DEFAULT_SWEEP,
    memory: bool = False,
    fmt: _ReportFormat = "rich",
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
        fmt: `rich` prints a coloured table to the terminal; `markdown` / `json` write a
            file per distribution to the output directory.
        output_dir: Where the `markdown` / `json` files are written. Defaults to `benchmarks/results/`.
    """
    require_release_build()
    names = distributions or list(REGISTRY)
    if unknown := [n for n in names if n not in REGISTRY]:
        msg = f"unknown distribution(s): {', '.join(unknown)}. Available: {', '.join(REGISTRY)}"
        raise ValueError(msg)
    results_dir = output_dir or (Path(__file__).parent / "results")

    for name in names:
        comp = REGISTRY[name]
        results: list[Result] = []
        try:
            for result in measure_cases(comp, sweep, memory=memory):
                results.append(result)  # noqa: PERF402 - an interrupt must leave the partial list intact
        finally:
            # `finally`, so an interrupt still reports the cells already measured before it stops.
            if results:
                emit(comp, results, sweep, fmt=fmt, output_dir=results_dir)


if __name__ == "__main__":
    app()
