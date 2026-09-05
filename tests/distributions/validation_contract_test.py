"""Every value-keyed method reports an invalid parameterisation, whichever branch the value selects."""

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
from tests._polars_compat import ARM_MASKING_HIDES_VALIDATION

if TYPE_CHECKING:
    from collections.abc import Callable

    from polars_stats.distributions._base import _UnivariateDistribution


@dataclass(frozen=True)
class _Case:
    """One distribution parameterised from columns, where row 0 is valid and row 1 is not.

    `fragment` is matched against the raised message, so it names the offending parameter.
    """

    name: str
    dist: _UnivariateDistribution
    columns: dict[str, list[float]]
    fragment: str


_CASES: tuple[_Case, ...] = (
    _Case("Bernoulli", Bernoulli(p=pl.col("p")), {"p": [0.3, 1.5]}, "p"),
    _Case("Beta", Beta(a=pl.col("a"), b=pl.col("b")), {"a": [2.0, -1.0], "b": [3.0, 3.0]}, "a"),
    _Case("Binomial", Binomial(n=pl.col("n"), p=pl.col("p")), {"n": [10, 10], "p": [0.3, 1.5]}, "p"),
    _Case("DiscreteUniform", DiscreteUniform(min=pl.col("lo"), max=pl.col("hi")), {"lo": [0, 5], "hi": [10, 2]}, "max"),
    _Case("Exponential", Exponential(rate=pl.col("r")), {"r": [1.0, -1.0]}, "rate"),
    _Case("Geometric", Geometric(p=pl.col("p")), {"p": [0.3, 1.5]}, "p"),
    _Case("LogNormal", LogNormal(mu=pl.col("m"), sigma=pl.col("s")), {"m": [0.0, 0.0], "s": [1.0, -1.0]}, "sigma"),
    _Case("Normal", Normal(mu=pl.col("m"), sigma=pl.col("s")), {"m": [0.0, 0.0], "s": [1.0, -1.0]}, "sigma"),
    _Case("Uniform", Uniform(min=pl.col("lo"), max=pl.col("hi")), {"lo": [0.0, 5.0], "hi": [1.0, 2.0]}, "max"),
)

_VALUE_METHODS = ("pdf", "log_pdf", "pmf", "log_pmf", "cdf", "log_cdf", "sf", "log_sf")
"""Support-point methods. `pdf` / `pmf` are family-specific, so a missing one is skipped, not failed."""

_QUANTILE_METHODS = ("ppf", "isf")

_SUPPORT_POINTS = (-3.0, -1.0, 0.0, 0.5, 1.0, 3.0, 100.0)
"""Spans below, inside and above every support, so each value-keyed branch gets at least one value."""

_QUANTILES = (-0.5, 0.001, 0.5, 0.999, 1.5)
"""In-range quantiles plus the two out-of-range values, which take the guard branch instead of computing.

A `None` quantile is deliberately *not* probed. Polars masks null rows out of an elementwise plugin's
input on every supported version, so the parameters on a null row never reach the validator and no
distribution raises, `DiscreteUniform` and the other Rust-backed ones included. That is a property of
plugin dispatch, not of the branch the value selects, so it says nothing about this contract.
"""


_LEAKING_METHODS: dict[str, frozenset[str]] = {
    "Bernoulli": frozenset({"pmf", "log_pmf", "cdf", "log_cdf", "sf", "log_sf", "ppf", "isf"}),
    "Exponential": frozenset({"pdf", "log_pdf", "cdf", "sf", "log_sf", "ppf", "isf"}),
    "Geometric": frozenset({"pmf", "log_pmf", "cdf", "log_cdf", "sf", "log_sf", "ppf", "isf"}),
    "Uniform": frozenset({"pdf", "log_pdf", "cdf", "log_cdf", "sf", "log_sf", "ppf", "isf"}),
}
"""The (distribution, method) pairs that `ARM_MASKING_HIDES_VALIDATION` is expected to break.

Measured, not assumed: these are exactly the pairs that fail on polars 1.44.1 and pass on 1.43.2.
`Exponential.log_cdf` is absent because it alone among the four distributions' 32 value-keyed methods
still raises, so gating it would `XPASS` under `xfail_strict`. Porting a distribution's closed forms
to Rust deletes its entry here; the last one to go deletes the dict and the constant with it.
"""

_METHOD_VALUES: tuple[tuple[str, tuple[float, ...]], ...] = (
    *((method, _SUPPORT_POINTS) for method in _VALUE_METHODS),
    *((method, _QUANTILES) for method in _QUANTILE_METHODS),
)
"""Each method paired with the whole value sweep it is contracted to raise on."""


def _ids(case: _Case) -> str:
    return case.name


def _report(case: _Case, method_fn: Callable[[pl.Expr], pl.Expr], value: float) -> str | None:
    """The `ComputeError` message the method raised at `value`, or `None` if it returned a result."""
    frame = pl.DataFrame({**case.columns, "v": pl.Series([value, value], dtype=pl.Float64)})
    try:
        frame.select(r=method_fn(pl.col("v")))
    except pl.exceptions.ComputeError as exc:
        return str(exc)
    return None


@pytest.mark.parametrize("case", _CASES, ids=_ids)
@pytest.mark.parametrize(("method", "values"), _METHOD_VALUES, ids=[method for method, _ in _METHOD_VALUES])
def test_invalid_parameter_raises_whichever_branch_the_value_selects(
    case: _Case, method: str, values: tuple[float, ...], request: pytest.FixtureRequest
) -> None:
    """One assertion per (distribution, method), over the whole value sweep.

    Which branch a value selects is an implementation detail of the method, so splitting the sweep
    into one test per value would make the pass/fail pattern an artifact of that detail rather than a
    statement about the contract. It would also make the `polars>=1.44` gate below inexact: leakage
    is not uniform across values within a method.
    """
    method_fn: Callable[[pl.Expr], pl.Expr] | None = getattr(case.dist, method, None)
    if method_fn is None:
        pytest.skip(f"{case.name} has no {method} (wrong family)")
    if ARM_MASKING_HIDES_VALIDATION and method in _LEAKING_METHODS.get(case.name, frozenset()):
        request.applymarker(pytest.mark.xfail(reason="pola-rs/polars#29005"))

    reports = [(value, _report(case, method_fn, value)) for value in values]
    # A `None` report means the invalid row in `columns` was silently computed rather than reported.
    accepted = [value for value, report in reports if report is None]
    assert not accepted, f"{case.name}.{method} accepted the invalid parameterisation at {accepted}"
    misnamed = [value for value, report in reports if report is not None and case.fragment not in report]
    assert not misnamed, f"{case.name}.{method} raised without naming `{case.fragment}` at {misnamed}"
