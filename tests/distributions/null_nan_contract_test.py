"""The parameter-validity half of the null / `NaN` contract: an invalid present parameter raises.

`NaN`, `+inf` and `-inf` in any float parameter slot raise `ComputeError` on every method, whatever
else the row holds: a `NaN` or null evaluation point, or a null sibling parameter. So does a finite
value outside the slot's own domain beside a null sibling. Validation is a property of the parameter
column, run in Rust once per call before any row is built, so it cannot depend on which *other*
columns happen to be null on the same row.

Value-keyed methods are probed through the private `_x` hooks: the public wrappers overlay
`propagate_null_and_nan` (`_base.py`), which from polars 1.44 masks the plugin out of the null and
`NaN` rows before it validates. Moments and samplers have no wrapper and go through the public API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import polars as pl
import pytest

from polars_stats import Bernoulli, Beta, Binomial, DiscreteUniform, Exponential, Geometric, LogNormal, Normal, Uniform
from polars_stats.distributions._base import DiscreteDistribution, _UnivariateDistribution


@dataclass(frozen=True)
class _Dist:
    """One distribution with every parameter a column, a valid row for it, and which slots are integers."""

    name: str
    dist: _UnivariateDistribution
    valid: dict[str, float]
    integer: frozenset[str] = frozenset()

    @property
    def float_slots(self) -> tuple[str, ...]:
        return tuple(name for name in self.valid if name not in self.integer)

    def frame(self, overrides: dict[str, float | None], x: float | None) -> pl.DataFrame:
        """One row: the valid parameters with `overrides` applied, and the evaluation point `x`."""
        schema = {name: (pl.Int64 if name in self.integer else pl.Float64) for name in self.valid}
        data = {name: [overrides.get(name, value)] for name, value in self.valid.items()}
        return pl.DataFrame({**data, "x": [x]}, schema={**schema, "x": pl.Float64})


_DISTS: tuple[_Dist, ...] = (
    _Dist("Bernoulli", Bernoulli(p=pl.col("p")), {"p": 0.5}),
    _Dist("Binomial", Binomial(n=pl.col("n"), p=pl.col("p")), {"n": 10, "p": 0.5}, integer=frozenset({"n"})),
    _Dist("Beta", Beta(a=pl.col("a"), b=pl.col("b")), {"a": 2.0, "b": 3.0}),
    _Dist("Exponential", Exponential(rate=pl.col("rate")), {"rate": 1.0}),
    _Dist("Geometric", Geometric(p=pl.col("p")), {"p": 0.5}),
    _Dist("Normal", Normal(mu=pl.col("mu"), sigma=pl.col("sigma")), {"mu": 0.0, "sigma": 1.0}),
    _Dist("LogNormal", LogNormal(mu=pl.col("mu"), sigma=pl.col("sigma")), {"mu": 0.0, "sigma": 1.0}),
    _Dist("Uniform", Uniform(min=pl.col("min"), max=pl.col("max")), {"min": 0.0, "max": 1.0}),
    _Dist(
        "DiscreteUniform",
        DiscreteUniform(min=pl.col("min"), max=pl.col("max")),
        {"min": 0, "max": 5},
        integer=frozenset({"min", "max"}),
    ),
)
_BY_NAME = {dist.name: dist for dist in _DISTS}

_MOMENTS = ("mean", "variance", "std", "median", "entropy")

_NON_FINITE = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}

_OUT_OF_DOMAIN: tuple[tuple[str, str, float], ...] = (
    ("Normal", "sigma", -1.0),
    ("LogNormal", "sigma", -1.0),
    ("Beta", "a", -1.0),
    ("Beta", "b", -1.0),
    ("Binomial", "p", -1.0),
)
"""A finite value outside a slot's own domain, for every two-parameter distribution that has one.

`Uniform` and `DiscreteUniform` have none: a bound alone is any finite value (any integer), and the
pair constraint runs only where both bounds are present.
"""


def _value_hooks(dist: _UnivariateDistribution) -> tuple[str, ...]:
    """Every value-keyed `_x` hook of `dist`, density methods named by family."""
    density = ("_pmf", "_log_pmf") if isinstance(dist, DiscreteDistribution) else ("_pdf", "_log_pdf")
    return (*density, "_cdf", "_log_cdf", "_sf", "_log_sf", "_ppf", "_isf")


def _report(frame: pl.DataFrame, expr: pl.Expr) -> str | None:
    """The `ComputeError` message, or `None` when the query computed."""
    try:
        frame.select(r=expr)
    except pl.exceptions.ComputeError as err:
        return str(err)
    return None


def _unreported(dist: _Dist, overrides: dict[str, float | None], slot: str) -> list[str]:
    """Every method that computes, or raises without naming `slot`, on the row `overrides` describes.

    The value-keyed hooks run at a finite, a `NaN` and a null evaluation point; the moments and the
    samplers read no point, so they run once.
    """
    probes = [
        (f"{hook}(x={x})", dist.frame(overrides, x), getattr(dist.dist, hook)(pl.col("x")))
        for x in (0.5, math.nan, None)
        for hook in _value_hooks(dist.dist)
    ]
    row = dist.frame(overrides, 0.5)
    probes += [(method, row, getattr(dist.dist, method)()) for method in _MOMENTS]
    probes += [("sample", row, dist.dist.sample(seed=0)), ("samples", row, dist.dist.samples(3, seed=0))]
    fragment = f"{slot} must be"
    return [
        label for label, frame, expr in probes if (report := _report(frame, expr)) is None or fragment not in report
    ]


_NON_FINITE_CASES = [
    pytest.param(dist, slot, bad, id=f"{dist.name}.{slot}={label}")
    for dist in _DISTS
    for slot in dist.float_slots
    for label, bad in _NON_FINITE.items()
]


@pytest.mark.parametrize(("dist", "slot", "bad"), _NON_FINITE_CASES)
def test_non_finite_parameter_raises_on_every_method(dist: _Dist, slot: str, bad: float) -> None:
    """With the sibling parameters present, and again with each sibling null."""
    rows: list[dict[str, float | None]] = [
        {slot: bad},
        *({slot: bad, other: None} for other in dist.valid if other != slot),
    ]
    unreported = [(label, row) for row in rows for label in _unreported(dist, row, slot)]
    assert not unreported, f"{dist.name} did not report `{slot} must be` in {unreported}"


@pytest.mark.parametrize(("name", "slot", "bad"), _OUT_OF_DOMAIN, ids=[f"{n}.{s}={b}" for n, s, b in _OUT_OF_DOMAIN])
def test_invalid_parameter_beside_a_null_sibling_raises(name: str, slot: str, bad: float) -> None:
    """A null sibling is a missing answer on its row, not a reason to skip validating the present parameter."""
    dist = _BY_NAME[name]
    (sibling,) = (other for other in dist.valid if other != slot)
    unreported = _unreported(dist, {slot: bad, sibling: None}, slot)
    assert not unreported, f"{name} did not report `{slot} must be` beside a null `{sibling}` in {unreported}"
