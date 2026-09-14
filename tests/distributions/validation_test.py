"""What every method does with an input that is not an ordinary in-domain number.

Four claims, one probe harness, every distribution and every method:

* an **invalid** parameter raises `ComputeError` naming the rule it broke, whatever the evaluation
  point holds and whatever else on the row is null;
* a **null** parameter nulls that row's answer and leaves its neighbours alone;
* a **null or `NaN` evaluation point** nulls or propagates, on valid parameters;
* a column whose dtype is not numeric is refused before any row is built, while `Decimal` computes as
  its `Float64` cast and a `Null`-typed column answers all-null.

Every probe runs on a **two-row frame whose first row is fully valid**, which is what a one-row
frame cannot see: a driver answering per *chunk* rather than per row passes a single-row probe.

The parameterisations come from `tests/_registry.invalid_cases`, which is the recorded finite
out-of-domain table plus the derived non-finite sweep, so the three tables that used to hold this and
had drifted apart are now one.
"""

from __future__ import annotations

import math
import operator
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl
import pytest

from polars_stats import Binomial, DiscreteUniform
from tests._polars_compat import ENGINE_SELECTABLE, assert_series_equal
from tests._registry import ALL_SPECS, DRIVER_REGIMES, MOMENTS, all_value_keyed, invalid_cases, point_column

if TYPE_CHECKING:
    from collections.abc import Callable

    from polars_stats.distributions._base import _UnivariateDistribution
    from tests._registry import DistSpec, InvalidCase, Regime

_Overrides = dict[str, float | None]


@dataclass(frozen=True)
class _Probe:
    """One method call, with the frame it reads. `label` is what a failure reports."""

    label: str
    frame: pl.DataFrame
    expr: pl.Expr
    valid_row_answers: bool = True
    """Whether row 0 must answer non-null. `False` only where the evaluation point is itself null."""

    def refusal(self) -> str | None:
        """The refusal as `<type>: <message>`, or `None` when the query computed."""
        try:
            self.frame.select(r=self.expr)
        except (pl.exceptions.ComputeError, pl.exceptions.InvalidOperationError) as err:
            return f"{type(err).__name__}: {err}"
        return None  # pragma: no cover

    def answers(self) -> list[object]:
        """The answer per row, `None` where the row nulled."""
        return self.frame.select(r=self.expr)["r"].to_list()


def _frame(spec: DistSpec, overrides: _Overrides, x: float | None) -> pl.DataFrame:
    """Two rows: the valid `example`, then the same with `overrides` applied. Both evaluated at `x`.

    The `q` column mirrors how ordinary `x` is rather than holding a fixed `0.5`, so `ppf` / `isf`
    meet the same `NaN` and null points as every other method. Pinning it at `0.5` left the two
    inverses as the only methods whose non-ordinary row was never built, which is exactly the
    `when`-arm hazard the invalid-parameter claim below exists to catch.
    """
    q = x if x is None or math.isnan(x) else 0.5
    schema = {p.column: p.dtype for p in spec.parameters}
    data: dict[str, list[float | None]] = {
        p.column: [p.cast(v), overrides.get(p.column, p.cast(v))] for p, v in spec.pairs()
    }
    return pl.DataFrame(
        {**data, "x": [x, x], "q": [q, q]},
        schema={**schema, "x": pl.Float64, "q": pl.Float64},
    )


def _probes(spec: DistSpec, overrides: _Overrides, points: tuple[float | None, ...]) -> list[_Probe]:
    """Every value-keyed method at every point, then the moments and both samplers."""
    dist = spec.build("name")
    probes = [
        _Probe(
            f"{method}({point_column(method)}={point})",
            _frame(spec, overrides, point),
            getattr(dist, method)(pl.col(point_column(method))),
            point is not None,
        )
        for point in points
        for method in all_value_keyed(dist)
    ]
    row = _frame(spec, overrides, spec.on_support_point)
    probes += [_Probe(method, row, getattr(dist, method)()) for method in MOMENTS]
    probes += [_Probe("sample", row, dist.sample(seed=0)), _Probe("samples", row, dist.samples(3, seed=0))]
    return probes


def _every_call(dist: _UnivariateDistribution) -> list[tuple[str, pl.Expr]]:
    """`(label, expression)` for every public method: the value-keyed ones, the moments, both samplers.

    The label is what a failure names, so a refusal that only some methods make points at the method
    rather than at the distribution.
    """
    calls: list[tuple[str, pl.Expr]] = [(m, getattr(dist, m)(pl.col(point_column(m)))) for m in all_value_keyed(dist)]
    calls += [(m, getattr(dist, m)()) for m in MOMENTS]
    return [*calls, ("sample", dist.sample(seed=0)), ("samples", dist.samples(3, seed=0))]


def _unreported(spec: DistSpec, overrides: _Overrides, case: InvalidCase) -> list[tuple[str, str | None]]:
    """Every method that computed, or raised without naming the broken rule, with what it reported."""
    reports = [(p.label, p.refusal()) for p in _probes(spec, overrides, (spec.on_support_point, math.nan, None))]
    return [(label, report) for label, report in reports if report is None or not re.search(case.match, report)]


_INVALID = [pytest.param(spec, case, id=f"{spec.name}.{case.label}") for spec, case in invalid_cases(ALL_SPECS)]
"""Every parameterisation that must be refused: the recorded finite ones, then the derived non-finite."""


@pytest.mark.parametrize(("spec", "case"), _INVALID)
def test_an_invalid_parameter_raises_on_every_method(spec: DistSpec, case: InvalidCase) -> None:
    """At a support point, at `NaN`, and at a null evaluation point alike.

    Validation runs over the parameter column before any row is built, so it cannot depend on what
    the evaluation point holds. A `when` arm is masked to null on the rows it does not select from
    polars 1.44, so a validator reachable only from inside an arm would never see the `NaN` row.
    """
    overrides: _Overrides = {p.column: bad for p, bad in zip(spec.parameters, case.params, strict=True)}
    unreported = _unreported(spec, overrides, case)
    assert not unreported, f"{spec.name} did not report `{case.match}`: {unreported}"


_WITH_NULL_SIBLING = [
    pytest.param(spec, case, other.column, id=f"{spec.name}.{case.label}+null-{other.name}")
    for spec, case in invalid_cases(ALL_SPECS)
    if (bad := spec.blamed(case)) is not None and not (bad.paired and case in spec.invalid)
    for other in spec.parameters
    if other is not bad
]
"""A `paired` parameter's *ordering* rule reads both bounds, so nulling one removes the comparison and
the row nulls: that is the contract, not a gap, and the recorded `Uniform` / `DiscreteUniform` rows
drop out here for it. Their **derived non-finite** rows stay: finiteness is judged per value, with no
sibling to compare, so `Uniform(min=NaN, max=null)` must still raise."""


@pytest.mark.parametrize(("spec", "case", "sibling"), _WITH_NULL_SIBLING)
def test_an_invalid_parameter_beside_a_null_sibling_still_raises(
    spec: DistSpec, case: InvalidCase, sibling: str
) -> None:
    """A null on the row does not downgrade an invalid parameterisation to a null answer."""
    overrides: _Overrides = {p.column: bad for p, bad in zip(spec.parameters, case.params, strict=True)}
    overrides[sibling] = None
    unreported = _unreported(spec, overrides, case)
    assert not unreported, f"{spec.name} did not report `{case.match}` beside a null `{sibling}`: {unreported}"


_SLOTS = [
    pytest.param(spec, param.column, id=f"{spec.name}.{param.name}") for spec in ALL_SPECS for param in spec.parameters
]


@pytest.mark.parametrize(("spec", "slot"), _SLOTS)
def test_a_null_parameter_nulls_that_row_only(spec: DistSpec, slot: str) -> None:
    """Every method nulls row 1 and answers row 0, at a finite, `NaN`, null and off-support point.

    The valid half carries as much as the null half: it stops a driver that nulls the whole chunk
    from passing, and stops an off-support constant from hiding a parameter that was never read.
    """
    points = (spec.on_support_point, math.nan, None, *spec.off_support_points)
    rows = [(probe, probe.answers()) for probe in _probes(spec, {slot: None}, points)]

    answered = [(probe.label, got[1]) for probe, got in rows if got[1] is not None]
    assert not answered, f"{spec.name} answered with a null `{slot}`: {answered}"
    swallowed = [probe.label for probe, got in rows if probe.valid_row_answers and got[0] is None]
    assert not swallowed, f"{spec.name} nulled the valid row too, with a null `{slot}`: {swallowed}"


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
def test_a_null_evaluation_point_nulls_the_answer(spec: DistSpec, regime: Regime) -> None:
    """A valid point beside a null one: the null nulls, the valid one still answers.

    Both parameter regimes are probed because the null check sits in each Rust driver separately:
    `value_keyed_scalar` for constant parameters, `value_keyed_binary` / `_ternary` for columns.
    """
    dist = spec.build(regime)
    frame = pl.DataFrame(
        {"x": [spec.on_support_point, None], "q": [0.5, None]},
        schema={"x": pl.Float64, "q": pl.Float64},
    )

    for method in all_value_keyed(dist):
        column = point_column(method)
        got = frame.select(r=getattr(dist, method)(pl.col(column)))["r"]
        assert got.len() == frame.height, f"{spec.name}.{method} returned {got.len()} rows"
        assert got[1] is None, f"{spec.name}.{method} answered {got[1]} at a null {column}"
        assert got[0] is not None, f"{spec.name}.{method} nulled the whole column, not just the null row"


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
def test_a_nan_evaluation_point_propagates_as_nan(spec: DistSpec, regime: Regime) -> None:
    """`NaN` in, `NaN` out: not null, and not a confident constant.

    The short-circuit is repeated in each Rust driver rather than shared, so both parameter regimes
    run. Two things go wrong without it: statrs' regularized incomplete beta *panics* on a `NaN`
    point and takes the whole query down, and `NaN < 0.0` is false while `NaN.floor() as u64`
    saturates to `0`, so an unguarded `Binomial` body returns a confident `P(X <= 0)`.
    """
    frame = pl.DataFrame(
        {"x": [spec.on_support_point, math.nan], "q": [0.5, math.nan]},
        schema={"x": pl.Float64, "q": pl.Float64},
    )
    dist = spec.build(regime)
    for method in all_value_keyed(dist):
        out = frame.select(r=getattr(dist, method)(pl.col(point_column(method))))["r"]
        assert out.null_count() == 0, f"{spec.name}-{regime}.{method} nulled a NaN evaluation point"
        assert not math.isnan(out[0]), f"{spec.name}-{regime}.{method} NaN-ed the ordinary row too"
        assert math.isnan(out[1]), f"{spec.name}-{regime}.{method} did not propagate NaN"


_REFUSED_DTYPES = (pl.Boolean(), pl.String(), pl.Date())
"""One per family polars' own cast would silently accept: `Boolean` as `0` / `1`, `String` parsed, `Date` as days."""


def _accepted(column: str, dtype: pl.DataType, probes: list[_Probe]) -> list[tuple[str, str]]:
    """Every probe that computes, or raises without naming `column`, with `column` recast to `dtype`.

    A moment may refuse with polars' own `InvalidOperationError` instead: its closed-form arithmetic
    (`n * p` on a `String` `p`) meets the column before the plugin does. Value-keyed methods and
    samplers reach the plugin first.
    """
    fragment = f"'{column}' must be a numeric column"
    refusals = [
        (p.label, _Probe(p.label, p.frame.with_columns(pl.col(column).cast(pl.Int64).cast(dtype)), p.expr).refusal())
        for p in probes
    ]
    return [
        (label, refusal or "computed")
        for label, refusal in refusals
        if refusal is None
        or (fragment not in refusal and not (label in MOMENTS and refusal.startswith("InvalidOperationError")))
    ]


def _dtype_probes(spec: DistSpec, column: str) -> list[_Probe]:
    """Every method the dtype gate stands in front of, all reading `column` as their evaluation point.

    When `column` is a parameter the samplers join in, since they read parameters but no point. The
    closed-form moments stay out of the *value* half: they are polars arithmetic on the parameter
    itself, so their dtype behaviour is polars', not the gate's.
    """
    dist = spec.build("name")
    frame = _frame(spec, {}, spec.on_support_point)
    probes = [_Probe(method, frame, getattr(dist, method)(pl.col("x"))) for method in all_value_keyed(dist)]
    if column != "x":
        probes += [_Probe("sample", frame, dist.sample(seed=0)), _Probe("samples", frame, dist.samples(3, seed=0))]
    return probes


@pytest.mark.parametrize("dtype", _REFUSED_DTYPES, ids=str)
def test_a_non_numeric_evaluation_point_raises_on_every_method(spec: DistSpec, dtype: pl.DataType) -> None:
    accepted = _accepted("x", dtype, _dtype_probes(spec, "x"))
    assert not accepted, f"{spec.name} took a {dtype} evaluation point: {accepted}"


_FLOAT_SLOTS = [
    pytest.param(spec, param.column, id=f"{spec.name}.{param.name}")
    for spec in ALL_SPECS
    for param in spec.float_parameters
]


@pytest.mark.parametrize("dtype", _REFUSED_DTYPES, ids=str)
@pytest.mark.parametrize(("spec", "slot"), _FLOAT_SLOTS)
def test_a_non_numeric_parameter_raises_on_every_method(spec: DistSpec, slot: str, dtype: pl.DataType) -> None:
    """Float slots only: `n` and `DiscreteUniform`'s bounds keep their own, stricter integer gate.

    The moments join the sweep here, since a bad parameter reaches them too; `_accepted` allows
    polars' own `InvalidOperationError` for those, whose arithmetic meets the column first.
    """
    probes = _dtype_probes(spec, slot)
    probes += [_Probe(m, _frame(spec, {}, spec.on_support_point), getattr(spec.build("name"), m)()) for m in MOMENTS]
    accepted = _accepted(slot, dtype, probes)
    assert not accepted, f"{spec.name} took a {dtype} `{slot}`: {accepted}"


_GATED_COLUMNS = [
    pytest.param(spec, column, id=f"{spec.name}.{column}")
    for spec in ALL_SPECS
    for column in ("x", *(p.column for p in spec.float_parameters))
]
"""Every column the gate sees: the evaluation point and each float parameter slot."""


@pytest.mark.parametrize(("spec", "column"), _GATED_COLUMNS)
def test_a_decimal_column_computes_as_its_float64_cast(spec: DistSpec, column: str) -> None:
    """The `Float64` side is rounded to the same scale first, so the two sides carry one value.

    Without that, `Decimal(10, 2)` would quietly re-round the input and the comparison would be
    against a different parameterisation rather than against a different dtype.
    """
    for probe in _dtype_probes(spec, column):
        rounded = probe.frame.with_columns(pl.col(column).round(2))
        decimal = rounded.with_columns(pl.col(column).cast(pl.Decimal(10, 2)))
        assert_series_equal(decimal.select(r=probe.expr)["r"], rounded.select(r=probe.expr)["r"], check_exact=True)


@pytest.mark.parametrize(("spec", "column"), _GATED_COLUMNS)
def test_a_null_dtype_column_nulls_every_answer(spec: DistSpec, column: str) -> None:
    """A `Null`-typed column passes the gate as all-null `Float64`."""
    answered = [
        probe.label
        for probe in _dtype_probes(spec, column)
        if any(a is not None for a in probe.frame.with_columns(pl.lit(None).alias(column)).select(r=probe.expr)["r"])
    ]
    assert not answered, f"{spec.name} answered with a Null-typed `{column}`: {answered}"


def test_a_scalar_and_a_column_parameterisation_refuse_the_same_thing(spec: DistSpec) -> None:
    """Both routings raise on every recorded invalid parameterisation, and name the same rule.

    `fast_path_test.py` makes the same comparison on one representative method per family; this one
    sweeps every public method, which is where a validator that only some methods call would show.
    """
    frame = pl.DataFrame({"x": [spec.on_support_point] * 2, "q": [0.5] * 2})
    unreported = [
        (f"{case.label}/{regime}/{method}", report or "computed")
        for case in spec.invalid
        for regime in DRIVER_REGIMES
        for method, expr in _every_call(spec.build(regime, case.params))
        if (report := _Probe(method, frame, expr).refusal()) is None or not re.search(case.match, report)
    ]
    assert not unreported, f"{spec.name} did not report the rule it broke: {unreported}"


# --------------------------------------------------------------------------------------------------
# The integer parameters. `Binomial.n` and both `DiscreteUniform` bounds carry a *stricter* gate than
# the numeric one above: any integer width is accepted and widened, a float column is refused even
# where casting it would be lossless, and the check is column-level rather than per row. Bespoke
# because only two distributions have such a parameter and their widening targets differ (`UInt64`
# against `Int64`), so a shared assertion would have to be written twice inside one function anyway.
# --------------------------------------------------------------------------------------------------

_NOT_INTEGER = "must be an integer column|bounds must be integer columns"
_INT_DTYPES = [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]


@pytest.mark.parametrize("n", [10.0, 10.5, -1.0])
def test_a_float_trial_count_column_is_refused_by_every_method(n: float) -> None:
    """One `select`, not one per method: a fix that made only `pmf` raise would leave `mean` and
    `variance` evaluating a different distribution from the same parameters."""
    frame = pl.DataFrame({"n": [10.0, n], "p": [0.5, 0.5]}, schema={"n": pl.Float64, "p": pl.Float64})
    dist = Binomial(pl.col("n"), pl.col("p"))
    with pytest.raises(pl.exceptions.ComputeError, match=_NOT_INTEGER):
        frame.select(pmf=dist.pmf(2.0), mean=dist.mean(), var=dist.variance(), draw=dist.sample(seed=0))


def test_an_explicit_cast_is_the_supported_route_for_a_float_trial_count() -> None:
    """The refusal message tells the caller to write this, so it has to hold, draws included."""
    data = {"p": [0.5, 0.25], "x": [0.0, 2.0]}
    native = pl.DataFrame({"n": [10, 3], **data}, schema_overrides={"n": pl.Int64})
    floats = pl.DataFrame({"n": [10.0, 3.0], **data}, schema_overrides={"n": pl.Float64})
    cast = Binomial(pl.col("n").cast(pl.Int64), pl.col("p"))
    direct = Binomial(pl.col("n"), pl.col("p"))

    calls: list[Callable[[Binomial], pl.Expr]] = [
        operator.methodcaller("pmf", pl.col("x")),
        operator.methodcaller("cdf", pl.col("x")),
        operator.methodcaller("mean"),
        operator.methodcaller("variance"),
        operator.methodcaller("entropy"),
        operator.methodcaller("sample", seed=0),
    ]
    for call in calls:
        assert_series_equal(floats.select(r=call(cast))["r"], native.select(r=call(direct))["r"], check_dtypes=True)


@pytest.mark.parametrize("dtype", _INT_DTYPES, ids=str)
def test_every_integer_width_is_accepted_for_a_trial_count(dtype: type[pl.DataType]) -> None:
    """Any integer width widens to `UInt64`, so the rule is integrality and not `Int64` specifically."""
    frame = pl.DataFrame({"n": [10], "p": [0.5], "x": [1.0]}, schema={"n": dtype, "p": pl.Float64, "x": pl.Float64})
    dist = Binomial(pl.col("n"), pl.col("p"))
    got = frame.select(pmf=dist.pmf(pl.col("x")), mean=dist.mean(), var=dist.variance())
    assert got.row(0) == pytest.approx((10 * 0.5**10, 5.0, 2.5))


@pytest.mark.parametrize("dtype", [pl.Int8, pl.Int16, pl.Int32, pl.Int64], ids=str)
def test_a_negative_trial_count_raises_at_every_signed_width(dtype: type[pl.DataType]) -> None:
    """A negative has no `u64` to widen to, so the strict widening is the sign check.

    Polars' own overflow message carries the offending value, so the error still names it.
    """
    frame = pl.DataFrame({"n": [5, -1], "p": [0.5, 0.5]}, schema={"n": dtype, "p": pl.Float64})
    with pytest.raises(pl.exceptions.ComputeError, match=r"conversion|non-negative") as raised:
        frame.select(r=Binomial(pl.col("n"), pl.col("p")).mean())
    assert "-1" in str(raised.value)


def test_a_negative_trial_count_is_judged_for_the_column_not_the_row() -> None:
    """`n` is checked once for the column where `p` is checked per row, so a null `p` cannot hide it."""
    negative = pl.DataFrame({"n": [-1], "p": [None]}, schema={"n": pl.Int64, "p": pl.Float64})
    with pytest.raises(pl.exceptions.ComputeError, match=r"conversion|non-negative"):
        negative.select(r=Binomial(pl.col("n"), pl.col("p")).mean())

    null_n = pl.DataFrame({"n": [None], "p": [0.5]}, schema={"n": pl.Int64, "p": pl.Float64})
    assert null_n.select(r=Binomial(pl.col("n"), pl.col("p")).mean())["r"].to_list() == [None]


def test_the_whole_u64_range_of_a_trial_count_is_accepted() -> None:
    """`n` reaches statrs as the `u64` it is there, so a count above `i64::MAX` is a count like any other.

    An `Int64` funnel mapped the upper half of the range to `null`. `mean` and `sample` only:
    `entropy` would sum the whole `{0, ..., n}` support, which the next test refuses outright.
    """
    trials = [2**63, 2**64 - 1]
    frame = pl.DataFrame({"n": trials, "p": [0.5, 0.5]}, schema={"n": pl.UInt64, "p": pl.Float64})
    dist = Binomial(n=pl.col("n"), p=pl.col("p"))

    got = frame.select(mean=dist.mean(), draw=dist.sample(seed=0))
    assert got["mean"].to_list() == pytest.approx([n * 0.5 for n in trials])
    assert got["draw"].dtype == pl.UInt64
    assert all(0 <= drawn <= n for drawn, n in zip(got["draw"].to_list(), trials, strict=True))


def test_the_binomial_entropy_raises_at_the_u64_maximum() -> None:
    """statrs iterates `(0..n + 1)`, which wraps to an empty range at `u64::MAX` in release.

    It returned a confident `0.0` where the true value is ~22.9 nats. The one value the test above
    cannot extend to entropy, refused loudly instead.
    """
    frame = pl.DataFrame({"n": [2**64 - 1], "p": [0.5]}, schema={"n": pl.UInt64, "p": pl.Float64})
    with pytest.raises(pl.exceptions.ComputeError, match="overflows the entropy support sum"):
        frame.select(r=Binomial(pl.col("n"), pl.col("p")).entropy())


def test_a_null_dtype_trial_count_column_propagates() -> None:
    """A `Null`-dtype column has no values the integer rule could protect, so it is null-in-null-out.

    `pmf(1.0)`, not `pmf(pl.lit(1.0))`: a Python scalar is row-aligned to the column length, where a
    caller-built length-1 literal expr would take a different broadcast path and hide the same
    truncation.

    The engine is pinned rather than left to `POLARS_ENGINE_AFFINITY`: only the in-memory engine
    exposes the truncation, the streaming engine broadcasts upstream and hides it, and streaming is
    the usual local default. Left unpinned, the one regression this test exists to catch would be
    visible only in CI.
    """
    frame = pl.DataFrame({"n": [None, None], "p": [0.5, 0.5]})  # n dtype: Null
    dist = Binomial(pl.col("n"), pl.col("p"))
    query = frame.lazy().select(mean=dist.mean(), pmf=dist.pmf(1.0), draw=dist.sample(seed=0))
    got = query.collect(engine="in-memory") if ENGINE_SELECTABLE else query.collect()
    assert got["mean"].to_list() == [None, None]
    assert got["pmf"].to_list() == [None, None]
    assert got["draw"].to_list() == [None, None]


def test_an_all_null_float_trial_count_column_still_raises_on_dtype() -> None:
    """The dtype rule is column-level: a float `n` raises even when every value in it is null."""
    frame = pl.DataFrame({"n": [None, None], "p": [0.5, 0.5]}, schema={"n": pl.Float64, "p": pl.Float64})
    with pytest.raises(pl.exceptions.ComputeError, match=_NOT_INTEGER):
        frame.select(r=Binomial(pl.col("n"), pl.col("p")).mean())


def test_a_float_discrete_uniform_bound_column_is_refused() -> None:
    """Column-level and judged before values: refused even where casting would be lossless here."""
    frame = pl.DataFrame({"lo": [0.0, 1.0], "hi": [5.0, 6.0]})
    with pytest.raises(pl.exceptions.ComputeError, match="bounds must be integer columns"):
        frame.select(r=DiscreteUniform(min=pl.col("lo"), max=pl.col("hi")).mean())


@pytest.mark.parametrize("dtype", [pl.Int8, pl.Int16, pl.Int32, pl.UInt8, pl.UInt32, pl.UInt64], ids=str)
def test_any_integer_discrete_uniform_bound_dtype_is_accepted(dtype: type[pl.DataType]) -> None:
    frame = pl.DataFrame({"lo": [1, 2], "hi": [6, 7]}, schema={"lo": dtype, "hi": dtype})
    got = frame.select(v=DiscreteUniform(min=pl.col("lo"), max=pl.col("hi")).mean())["v"]
    assert got.to_list() == [3.5, 4.5]


@pytest.mark.parametrize("dtype", [pl.Int8, pl.Int16], ids=str)
def test_a_support_wider_than_the_bound_dtype_does_not_wrap(dtype: type[pl.DataType]) -> None:
    """Bound arithmetic widens to `Int64`, so a support wider than the bounds' own dtype stays exact."""
    frame = pl.DataFrame({"lo": [-100], "hi": [100], "x": [99]}, schema={"lo": dtype, "hi": dtype, "x": dtype})
    dist = DiscreteUniform(min=pl.col("lo"), max=pl.col("hi"))
    got = frame.select(
        mean=dist.mean(),
        median=dist.median(),
        cdf=dist.cdf(pl.col("x")),
        sf=dist.sf(pl.col("x")),
        log_cdf=dist.log_cdf(pl.col("x")),
        log_sf=dist.log_sf(pl.col("x")),
    )
    assert got["mean"][0] == 0.0
    assert got["median"][0] == 0.0
    assert got["cdf"][0] == pytest.approx(200 / 201)
    assert got["sf"][0] == pytest.approx(1 / 201)
    assert got["log_cdf"][0] == pytest.approx(math.log(200 / 201))
    assert got["log_sf"][0] == pytest.approx(math.log(1 / 201))


def test_an_unsigned_discrete_uniform_bound_above_int64_max_raises() -> None:
    """A `UInt64` column holds values `Int64` cannot; the strict cast reports them rather than wrapping."""
    frame = pl.DataFrame({"lo": [0, 1], "hi": [2**63 + 5, 20]}, schema={"lo": pl.UInt64, "hi": pl.UInt64})
    with pytest.raises(pl.exceptions.ComputeError, match="bounds must be integers that fit in i64"):
        frame.select(r=DiscreteUniform(min=pl.col("lo"), max=pl.col("hi")).mean())


def test_a_discrete_uniform_support_width_overflowing_int64_raises() -> None:
    with pytest.raises(pl.exceptions.ComputeError, match="support width"):
        pl.DataFrame({"x": [0.0]}).select(r=DiscreteUniform(min=-(2**63), max=2**63 - 1).mean())


@pytest.mark.parametrize("null_bound", ["lo", "hi"])
def test_a_null_dtype_discrete_uniform_bound_column_propagates_null(null_bound: str) -> None:
    schema = {"lo": pl.Int64, "hi": pl.Int64} | {null_bound: pl.Null}
    frame = pl.DataFrame({"lo": [1, 2], "hi": [6, 7]} | {null_bound: [None, None]}, schema=schema)
    got = frame.select(v=DiscreteUniform(min=pl.col("lo"), max=pl.col("hi")).support_size)["v"]
    assert got.to_list() == [None, None]


@pytest.mark.parametrize(("lo", "hi", "expected"), [(1, 6, 6.0), (-5, 9, 15.0), (3, 3, 1.0), (-20, -10, 11.0)])
def test_the_discrete_uniform_support_size_counts_both_bounds(lo: int, hi: int, expected: float) -> None:
    """`max - min + 1`: both bounds inclusive, so a one-point mass counts 1 rather than 0."""
    got = pl.DataFrame({"_": [0]}).select(v=DiscreteUniform(min=lo, max=hi).support_size).item(0, "v")
    assert got == expected


def test_an_invalid_constant_parameterisation_raises_without_a_frame() -> None:
    """`pl.select` resolves `pl.len()` to 0, so a method anchored to the frame would report nothing.

    Constant parameters are validated by their own length-1 plugin call, so every method must raise
    even where there is no frame to iterate.
    """
    dist = DiscreteUniform(min=5, max=0)
    for expr in (dist.pmf(2.0), dist.cdf(2.0), dist.ppf(0.5), dist.mean(), dist.support_size, dist.sample(seed=0)):
        with pytest.raises(pl.exceptions.ComputeError, match="max must be"):
            pl.select(r=expr)
