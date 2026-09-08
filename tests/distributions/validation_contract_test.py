"""Every value-keyed method reports an invalid parameterisation, whatever the evaluation point holds.

The first test sweeps the *evaluation value* across every branch of every method, which holds because
every value-keyed method validates inside the Rust plugin that computes it. The other two hold the
value at `NaN` and at null, the rows a validator inside a `when` arm would never see.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from collections.abc import Callable

    from polars_stats.distributions._base import _UnivariateDistribution


@dataclass(frozen=True)
class _Case:
    """One distribution parameterised from columns, where row 0 is valid and row 1 is not.

    `fragment` is matched against the raised message, so it names the offending parameter. It carries
    enough of the validator's own wording to fail on the wrong parameter: a bare `"p"` matches the
    `the plugin failed with message:` preamble polars wraps every plugin error in, which would make
    the check unfailable for the four distributions whose parameter is one letter.
    """

    name: str
    dist: _UnivariateDistribution
    columns: dict[str, list[float]]
    fragment: str


_CASES: tuple[_Case, ...] = (
    _Case("Bernoulli", Bernoulli(p=pl.col("p")), {"p": [0.3, 1.5]}, "p must be in"),
    _Case("Beta", Beta(a=pl.col("a"), b=pl.col("b")), {"a": [2.0, -1.0], "b": [3.0, 3.0]}, "a must be"),
    _Case("Binomial", Binomial(n=pl.col("n"), p=pl.col("p")), {"n": [10, 10], "p": [0.3, 1.5]}, "p must be in"),
    _Case(
        "DiscreteUniform",
        DiscreteUniform(min=pl.col("lo"), max=pl.col("hi")),
        {"lo": [0, 5], "hi": [10, 2]},
        "max must be",
    ),
    _Case("Exponential", Exponential(rate=pl.col("r")), {"r": [1.0, -1.0]}, "rate must be"),
    _Case("Geometric", Geometric(p=pl.col("p")), {"p": [0.3, 1.5]}, "p must be in"),
    _Case(
        "LogNormal", LogNormal(mu=pl.col("m"), sigma=pl.col("s")), {"m": [0.0, 0.0], "s": [1.0, -1.0]}, "sigma must be"
    ),
    _Case("Normal", Normal(mu=pl.col("m"), sigma=pl.col("s")), {"m": [0.0, 0.0], "s": [1.0, -1.0]}, "sigma must be"),
    _Case("Uniform", Uniform(min=pl.col("lo"), max=pl.col("hi")), {"lo": [0.0, 5.0], "hi": [1.0, 2.0]}, "max must be"),
)

_SUPPORT_POINT_METHODS = ("pdf", "log_pdf", "pmf", "log_pmf", "cdf", "log_cdf", "sf", "log_sf")
"""Support-point methods. `pdf` / `pmf` are family-specific, so a missing one is skipped, not failed."""

_QUANTILE_METHODS = ("ppf", "isf")

_SUPPORT_POINTS = (-3.0, -1.0, 0.0, 0.5, 1.0, 3.0, 100.0)
"""Spans below, inside and above every support, so each value-keyed branch gets at least one value."""

_QUANTILES = (-0.5, 0.001, 0.5, 0.999, 1.5)
"""In-range quantiles plus the two out-of-range values, which take the guard branch instead of computing.

A `None` quantile is not probed here. It is not a statement about which branch the value selects, so
it gets its own test below, alongside the `NaN` point.
"""


_METHOD_VALUES: tuple[tuple[str, tuple[float, ...]], ...] = (
    *((method, _SUPPORT_POINTS) for method in _SUPPORT_POINT_METHODS),
    *((method, _QUANTILES) for method in _QUANTILE_METHODS),
)
"""Each method paired with the whole value sweep it is contracted to raise on."""


def _ids(case: _Case) -> str:
    return case.name


def _report(case: _Case, method_fn: Callable[[pl.Expr], pl.Expr], value: float | None) -> str | None:
    """The `ComputeError` message the method raised at `value`, or `None` if it returned a result."""
    frame = pl.DataFrame({**case.columns, "v": pl.Series([value, value], dtype=pl.Float64)})
    try:
        frame.select(r=method_fn(pl.col("v")))
    except pl.exceptions.ComputeError as exc:
        return str(exc)


@pytest.mark.parametrize("case", _CASES, ids=_ids)
@pytest.mark.parametrize(("method", "values"), _METHOD_VALUES, ids=[method for method, _ in _METHOD_VALUES])
def test_invalid_parameter_raises_whichever_branch_the_value_selects(
    case: _Case, method: str, values: tuple[float, ...]
) -> None:
    """One assertion per (distribution, method), over the whole value sweep.

    Which branch a value selects is an implementation detail of the method, so splitting the sweep
    into one test per value would make the pass/fail pattern an artifact of that detail rather than a
    statement about the contract.
    """
    method_fn: Callable[[pl.Expr], pl.Expr] | None = getattr(case.dist, method, None)
    if method_fn is None:
        pytest.skip(f"{case.name} has no {method} (wrong family)")

    reports = [(value, _report(case, method_fn, value)) for value in values]
    # A `None` report means the invalid row in `columns` was silently computed rather than reported.
    accepted = [value for value, report in reports if report is None]
    assert not accepted, f"{case.name}.{method} accepted the invalid parameterisation at {accepted}"
    misnamed = [value for value, report in reports if report is not None and case.fragment not in report]
    assert not misnamed, f"{case.name}.{method} raised without naming `{case.fragment}` at {misnamed}"


_VALUE_KEYED_METHODS = (*_SUPPORT_POINT_METHODS, *_QUANTILE_METHODS)
"""All ten; the `NaN` and null points below are probed on every one."""


def _unreported_at(case: _Case, point: float | None) -> list[str]:
    """Every method that computes at `point`, or raises without naming `case.fragment`."""
    probed = [(method, fn) for method in _VALUE_KEYED_METHODS if (fn := getattr(case.dist, method, None)) is not None]
    # Eight of the ten always resolve; the other two are the wrong-family `pdf` / `pmf` pair. Asserted
    # so a renamed method shrinks the sweep loudly instead of silently.
    assert len(probed) == len(_VALUE_KEYED_METHODS) - 2
    reports = [(method, _report(case, fn, point)) for method, fn in probed]
    # Silence and a misnamed message both mean the invalid row went unreported.
    return [method for method, report in reports if report is None or case.fragment not in report]


@pytest.mark.parametrize("case", _CASES, ids=_ids)
def test_invalid_parameter_raises_at_a_nan_evaluation_point(case: _Case) -> None:
    """One assertion per distribution, over every value-keyed method, at a `NaN` evaluation point.

    Looped, not parametrised: the point is the same for every method, so one item per distribution is
    the claim.

    From polars 1.44 a `when` arm is masked to null on the rows it does not select, so a validator
    reachable only from inside an arm never sees a `NaN` row; the plugin has to answer that row itself.
    """
    unreported = _unreported_at(case, float("nan"))
    assert not unreported, f"{case.name} did not report the invalid `{case.fragment}` at a NaN point in {unreported}"


@pytest.mark.parametrize("case", _CASES, ids=_ids)
def test_invalid_parameter_raises_at_a_null_evaluation_point(case: _Case) -> None:
    """The null sibling of the test above: an invalid parameterisation raises whatever the value is.

    A null *value* propagates to null, but it never downgrades an invalid parameterisation to one.
    That is what separates it from a null *parameter*, where there is nothing to reject and the row
    nulls. Every per-row driver validates before it reads the value.
    """
    unreported = _unreported_at(case, None)
    assert not unreported, f"{case.name} did not report the invalid `{case.fragment}` at a null point in {unreported}"
