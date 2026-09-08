"""An invalid present parameter raises, whatever else its row holds; a null one nulls every answer.

`NaN`, `+inf` and `-inf` in any float parameter slot raise `ComputeError` on every method, beside a
`NaN` or null evaluation point and beside a null sibling parameter alike; so does a finite value
outside the slot's domain beside a null sibling. Validation runs over the parameter column before any
row is built, so it cannot depend on what else is null on the row.

A null parameter is answered before the evaluation point is read, so every method nulls at a finite,
`NaN`, null and off-support point alike.

A column that is not numeric (`Int*`, `UInt*`, `Float*`, `Decimal`, or `Null`-typed) is refused in either
position before any row is built, with a `ComputeError` naming the column and its dtype; `Decimal` computes
as its `Float64` cast and a `Null`-typed column as all-null.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import polars as pl
import pytest

from polars_stats import (
    Bernoulli,
    Beta,
    Binomial,
    DiscreteUniform,
    Exponential,
    Geometric,
    LogNormal,
    Normal,
    Uniform,
)
from polars_stats.distributions._base import DiscreteDistribution, _UnivariateDistribution
from tests._polars_compat import assert_series_equal


@dataclass(frozen=True)
class _Dist:
    name: str
    dist: _UnivariateDistribution
    valid: dict[str, float]
    integer: frozenset[str] = frozenset()
    off_support: tuple[float, ...] = ()
    """Points off the support; empty where the support is the whole line."""

    @property
    def float_slots(self) -> tuple[str, ...]:
        return tuple(name for name in self.valid if name not in self.integer)

    def frame(self, overrides: dict[str, float | None], x: float | None) -> pl.DataFrame:
        """One row: the valid parameters with `overrides` applied, and the evaluation point `x`."""
        schema = {name: (pl.Int64 if name in self.integer else pl.Float64) for name in self.valid}
        data = {name: [overrides.get(name, value)] for name, value in self.valid.items()}
        return pl.DataFrame({**data, "x": [x]}, schema={**schema, "x": pl.Float64})


_DISTS: tuple[_Dist, ...] = (
    _Dist("Bernoulli", Bernoulli(p=pl.col("p")), {"p": 0.5}, off_support=(2.0,)),
    _Dist(
        "Binomial",
        Binomial(n=pl.col("n"), p=pl.col("p")),
        {"n": 10, "p": 0.5},
        integer=frozenset({"n"}),
        off_support=(-1.0,),
    ),
    _Dist("Beta", Beta(a=pl.col("a"), b=pl.col("b")), {"a": 2.0, "b": 3.0}, off_support=(2.0,)),
    _Dist("Exponential", Exponential(rate=pl.col("rate")), {"rate": 1.0}, off_support=(-1.0,)),
    _Dist("Geometric", Geometric(p=pl.col("p")), {"p": 0.5}, off_support=(0.0, 2.5)),
    _Dist("Normal", Normal(mu=pl.col("mu"), sigma=pl.col("sigma")), {"mu": 0.0, "sigma": 1.0}),
    _Dist(
        "LogNormal", LogNormal(mu=pl.col("mu"), sigma=pl.col("sigma")), {"mu": 0.0, "sigma": 1.0}, off_support=(-1.0,)
    ),
    _Dist("Uniform", Uniform(min=pl.col("min"), max=pl.col("max")), {"min": 0.0, "max": 1.0}, off_support=(5.0,)),
    _Dist(
        "DiscreteUniform",
        DiscreteUniform(min=pl.col("min"), max=pl.col("max")),
        {"min": 0, "max": 5},
        integer=frozenset({"min", "max"}),
        off_support=(10.0,),
    ),
)
_BY_NAME = {dist.name: dist for dist in _DISTS}

_MOMENTS = ("mean", "variance", "std", "median", "entropy")
_NON_FINITE = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}

# `Uniform` is absent: a bound alone is any finite value, and the pair rule needs both bounds present.
_OUT_OF_DOMAIN: tuple[tuple[str, str, float], ...] = (
    ("Normal", "sigma", -1.0),
    ("LogNormal", "sigma", -1.0),
    ("Beta", "a", -1.0),
    ("Beta", "b", -1.0),
    ("Binomial", "p", -1.0),
)


def _value_keyed_methods(dist: _UnivariateDistribution) -> tuple[str, ...]:
    density = ("pmf", "log_pmf") if isinstance(dist, DiscreteDistribution) else ("pdf", "log_pdf")
    return (*density, "cdf", "log_cdf", "sf", "log_sf", "ppf", "isf")


def _refusal(frame: pl.DataFrame, expr: pl.Expr) -> str | None:
    """The refusal as `<type>: <message>`, or `None` when the query computed."""
    try:
        frame.select(r=expr)
    except (pl.exceptions.ComputeError, pl.exceptions.InvalidOperationError) as err:
        return f"{type(err).__name__}: {err}"


_Probes = list[tuple[str, pl.DataFrame, pl.Expr]]
_Overrides = dict[str, float | None]


def _value_keyed_probes(dist: _Dist, overrides: _Overrides, points: tuple[float | None, ...]) -> _Probes:
    """Every value-keyed method at every point, as `(label, frame, expr)`."""
    return [
        (f"{method}(x={x})", dist.frame(overrides, x), getattr(dist.dist, method)(pl.col("x")))
        for x in points
        for method in _value_keyed_methods(dist.dist)
    ]


def _sampler_probes(dist: _Dist, overrides: _Overrides) -> _Probes:
    row = dist.frame(overrides, 0.5)
    return [("sample", row, dist.dist.sample(seed=0)), ("samples", row, dist.dist.samples(3, seed=0))]


def _probes(dist: _Dist, overrides: _Overrides, points: tuple[float | None, ...]) -> _Probes:
    """The value-keyed methods at every point, then the moments and both samplers."""
    row = dist.frame(overrides, 0.5)
    moments = [(method, row, getattr(dist.dist, method)()) for method in _MOMENTS]
    return _value_keyed_probes(dist, overrides, points) + moments + _sampler_probes(dist, overrides)


def _unreported(dist: _Dist, overrides: _Overrides, slot: str) -> list[tuple[str, str | None]]:
    """Every method that computes, or raises without naming `slot`, with what it reported."""
    fragment = f"{slot} must be"
    reports = [(label, _refusal(frame, expr)) for label, frame, expr in _probes(dist, overrides, (0.5, math.nan, None))]
    return [(label, report) for label, report in reports if report is None or fragment not in report]


@pytest.mark.parametrize(
    ("dist", "slot", "bad"),
    [
        pytest.param(dist, slot, bad, id=f"{dist.name}.{slot}={label}")
        for dist in _DISTS
        for slot in dist.float_slots
        for label, bad in _NON_FINITE.items()
    ],
)
def test_non_finite_parameter_raises_on_every_method(dist: _Dist, slot: str, bad: float) -> None:
    """With the sibling parameters present, and again with each sibling null."""
    rows: list[dict[str, float | None]] = [
        {slot: bad},
        *({slot: bad, other: None} for other in dist.valid if other != slot),
    ]
    unreported = [(label, row, report) for row in rows for label, report in _unreported(dist, row, slot)]
    assert not unreported, f"{dist.name} did not report `{slot} must be`: {unreported}"


@pytest.mark.parametrize(("name", "slot", "bad"), _OUT_OF_DOMAIN, ids=[f"{n}.{s}={b}" for n, s, b in _OUT_OF_DOMAIN])
def test_invalid_parameter_beside_a_null_sibling_raises(name: str, slot: str, bad: float) -> None:
    dist = _BY_NAME[name]
    (sibling,) = (other for other in dist.valid if other != slot)
    unreported = _unreported(dist, {slot: bad, sibling: None}, slot)
    assert not unreported, f"{name} did not report `{slot} must be` beside a null `{sibling}`: {unreported}"


def _non_null_answers(dist: _Dist, slot: str) -> list[str]:
    """Every method that answers something other than null with `slot` null and the rest valid."""
    probes = _probes(dist, {slot: None}, (0.5, math.nan, None, *dist.off_support))
    return [label for label, frame, expr in probes if frame.select(r=expr)["r"].item() is not None]


@pytest.mark.parametrize(
    ("dist", "slot"),
    [pytest.param(dist, slot, id=f"{dist.name}.{slot}") for dist in _DISTS for slot in dist.valid],
)
def test_null_parameter_nulls_every_method(dist: _Dist, slot: str) -> None:
    """Off-support constants and the `NaN` short-circuit included."""
    answered = _non_null_answers(dist, slot)
    assert not answered, f"{dist.name} answered with a null `{slot}`: {answered}"


_REFUSED_DTYPES = (pl.Boolean(), pl.String(), pl.Date())
"""One per family polars' own cast would silently accept: `Boolean` as `0` / `1`, `String` parsed, `Date` as days."""


def _accepted(column: str, dtype: pl.DataType, probes: _Probes) -> list[tuple[str, str]]:
    """Every probe that computes, or raises without naming `column`, with `column` recast to `dtype`.

    A moment may refuse with polars' own `InvalidOperationError` instead: its closed-form arithmetic (`n * p`
    on a `String` `p`) meets the column before the plugin does. Value-keyed methods and samplers reach the plugin
    first.
    """
    fragment = f"'{column}' must be a numeric column"
    refusals = [
        (label, _refusal(frame.with_columns(pl.col(column).cast(pl.Int64).cast(dtype)), expr))
        for label, frame, expr in probes
    ]
    return [
        (label, refusal or "computed")
        for label, refusal in refusals
        if refusal is None
        or (fragment not in refusal and not (label in _MOMENTS and refusal.startswith("InvalidOperationError")))
    ]


@pytest.mark.parametrize("dtype", _REFUSED_DTYPES, ids=str)
@pytest.mark.parametrize("dist", _DISTS, ids=lambda dist: dist.name)
def test_non_numeric_evaluation_point_raises_on_every_method(dist: _Dist, dtype: pl.DataType) -> None:
    accepted = _accepted("x", dtype, _value_keyed_probes(dist, {}, (0.5,)))
    assert not accepted, f"{dist.name} took a {dtype} evaluation point: {accepted}"


@pytest.mark.parametrize("dtype", _REFUSED_DTYPES, ids=str)
@pytest.mark.parametrize(
    ("dist", "slot"),
    [pytest.param(dist, slot, id=f"{dist.name}.{slot}") for dist in _DISTS for slot in dist.float_slots],
)
def test_non_numeric_parameter_raises_on_every_method(dist: _Dist, slot: str, dtype: pl.DataType) -> None:
    """Float slots only: `n` and `DiscreteUniform`'s bounds keep their own, stricter integer gate."""
    accepted = _accepted(slot, dtype, _probes(dist, {}, (0.5,)))
    assert not accepted, f"{dist.name} took a {dtype} `{slot}`: {accepted}"


_COLUMNS = [
    pytest.param(dist, column, id=f"{dist.name}.{column}") for dist in _DISTS for column in ("x", *dist.float_slots)
]
"""Every column the gate sees: the evaluation point and each float parameter slot."""


def _gate_probes(dist: _Dist, column: str) -> _Probes:
    """The probes where only the gate stands between `column` and the answer.

    The closed-form moments are left out: they are polars arithmetic on the parameter itself, so their dtype
    behaviour is polars' (a missing kernel raises, a bare parameter keeps its dtype), not the gate's.
    """
    return _value_keyed_probes(dist, {}, (0.5,)) + ([] if column == "x" else _sampler_probes(dist, {}))


@pytest.mark.parametrize(("dist", "column"), _COLUMNS)
def test_decimal_column_computes_as_its_float64_cast(dist: _Dist, column: str) -> None:
    for _label, frame, expr in _gate_probes(dist, column):
        decimal = frame.with_columns(pl.col(column).cast(pl.Decimal(10, 2)))
        assert_series_equal(decimal.select(r=expr)["r"], frame.select(r=expr)["r"], check_exact=True)


@pytest.mark.parametrize(("dist", "column"), _COLUMNS)
def test_null_dtype_column_nulls_every_answer(dist: _Dist, column: str) -> None:
    """A `Null`-typed column passes the gate as all-null `Float64`."""
    answered = [
        label
        for label, frame, expr in _gate_probes(dist, column)
        if frame.with_columns(pl.lit(None).alias(column)).select(r=expr)["r"].item() is not None
    ]
    assert not answered, f"{dist.name} answered with a Null-typed `{column}`: {answered}"
