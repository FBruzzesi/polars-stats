"""Every value-keyed method reports an invalid parameterisation, whichever branch the value selects.

Two axes, one contract. The first test sweeps the *evaluation value* across every branch of every
method, and is what the Rust ports exist to make green. The second holds the value at `NaN` and
targets `propagate_null_and_nan` instead, a `when/then/otherwise` wrapped around every public
value-keyed method: a second place the validator can sit inside an arm, one level above the
distribution.
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
from tests._polars_compat import ARM_MASKING_HIDES_VALIDATION

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
    _Case("Beta", Beta(a=pl.col("a"), b=pl.col("b")), {"a": [2.0, -1.0], "b": [3.0, 3.0]}, "a and b must be"),
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

_VALUE_METHODS = ("pdf", "log_pdf", "pmf", "log_pmf", "cdf", "log_cdf", "sf", "log_sf")
"""Support-point methods. `pdf` / `pmf` are family-specific, so a missing one is skipped, not failed."""

_QUANTILE_METHODS = ("ppf", "isf")

_SUPPORT_POINTS = (-3.0, -1.0, 0.0, 0.5, 1.0, 3.0, 100.0)
"""Spans below, inside and above every support, so each value-keyed branch gets at least one value."""

_QUANTILES = (-0.5, 0.001, 0.5, 0.999, 1.5)
"""In-range quantiles plus the two out-of-range values, which take the guard branch instead of computing.

A `None` quantile is not probed here. It is not a statement about which branch the value selects, so
it gets its own test below, alongside the `NaN` point that leaks through the same wrapper.
"""


_LEAKING_METHODS: dict[str, frozenset[str]] = {
    "Geometric": frozenset({"pmf", "log_pmf", "cdf", "log_cdf", "sf", "log_sf", "ppf", "isf"}),
}
"""The (distribution, method) pairs that `ARM_MASKING_HIDES_VALIDATION` is expected to break.

Measured, not assumed: these are exactly the pairs that fail on polars 1.44.1 and pass on 1.43.2.
Every value-keyed method of the one remaining distribution leaks. Porting a distribution's closed
forms to Rust deletes its entry here; the last one to go deletes the dict.
"""

_METHOD_VALUES: tuple[tuple[str, tuple[float, ...]], ...] = (
    *((method, _SUPPORT_POINTS) for method in _VALUE_METHODS),
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


_NAN_METHODS = (*_VALUE_METHODS, *_QUANTILE_METHODS)
"""Every value-keyed method, since the wrapper below sits on all of them identically."""


@pytest.mark.parametrize("case", _CASES, ids=_ids)
def test_invalid_parameter_raises_at_a_nan_evaluation_point(case: _Case, request: pytest.FixtureRequest) -> None:
    """One assertion per distribution, over every value-keyed method, at a `NaN` evaluation point.

    The methods are looped inside rather than parametrised over, unlike the test above: leakage there
    varies by method, so its gate needs method precision, while here all 72 (distribution, method)
    pairs leak on polars 1.44.1 and all 72 raise on 1.43.2. One item per distribution matches that
    shape and keeps the gate to one marker instead of seventy-two.

    The leak is in `propagate_null_and_nan` (`_base.py`), not in any distribution: it spells the
    null/`NaN` overlay as `when(...).then(...).otherwise(result)`, so from polars 1.44 the plugin in
    `result` is masked out on exactly the `NaN` rows and never validates. Bypassing the wrapper makes
    the same call raise, which is why `Bernoulli` and `DiscreteUniform` leak here despite computing
    entirely in Rust. Porting a distribution does not fix this one; deleting the wrapper does.
    """
    # Passed as the marker's own condition rather than an `if`, since it is true for every item here:
    # a Python-level branch would leave its false side unexecuted on polars >= 1.44.
    reason = "pola-rs/polars#29005, at the wrapper rather than the hook"
    request.applymarker(pytest.mark.xfail(ARM_MASKING_HIDES_VALIDATION, reason=reason))

    probed = [(method, fn) for method in _NAN_METHODS if (fn := getattr(case.dist, method, None)) is not None]
    # Eight of the ten always resolve; the other two are the wrong-family `pdf` / `pmf` pair. Asserted
    # so a renamed method shrinks the sweep loudly instead of silently.
    assert len(probed) == len(_NAN_METHODS) - 2
    reports = [(method, _report(case, fn, float("nan"))) for method, fn in probed]
    # Silence and a misnamed message are merged into one predicate, where the test above keeps them
    # apart. Every probe here leaks on the gated version, so a second assertion would be a line no
    # supported polars ever reaches.
    unreported = [method for method, report in reports if report is None or case.fragment not in report]
    assert not unreported, f"{case.name} did not report the invalid `{case.fragment}` at a NaN point in {unreported}"


_RUST_CASES = tuple(case for case in _CASES if case.name not in _LEAKING_METHODS)
"""The cases whose value-keyed methods compute in Rust, derived rather than listed again.

A distribution leaves `_LEAKING_METHODS` exactly when its closed forms move into Rust, so this set
grows on its own as the port series lands.
"""


@pytest.mark.parametrize("case", _RUST_CASES, ids=_ids)
def test_invalid_parameter_raises_at_a_null_evaluation_point(case: _Case) -> None:
    """The null sibling of the test above: an invalid parameterisation raises whatever the value is.

    A null *value* propagates to null, but it never downgrades an invalid parameterisation to one.
    That is what separates it from a null *parameter*, where there is nothing to reject and the row
    nulls. Every per-row driver validates before it reads the value, so this holds on every supported
    polars, which is why it needs no gate where the two tests above do.

    Probed through the private hooks: `propagate_null_and_nan` spells the null overlay as the same
    `when(...).then(...).otherwise(result)` the `NaN` test above indicts, so the public wrappers still
    mask this from polars 1.44. Deleting the wrapper is what closes that half.
    """
    probed = [(method, fn) for method in _NAN_METHODS if (fn := getattr(case.dist, f"_{method}", None)) is not None]
    assert len(probed) == len(_NAN_METHODS) - 2
    reports = [(method, _report(case, fn, None)) for method, fn in probed]
    unreported = [method for method, report in reports if report is None or case.fragment not in report]
    assert not unreported, f"{case.name} did not report the invalid `{case.fragment}` at a null point in {unreported}"
