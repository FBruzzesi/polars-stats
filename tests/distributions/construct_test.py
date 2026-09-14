"""What the constructor checks, and what it deliberately leaves to evaluation.

`_coerce` is one function and its rule is the same for every parameter of every distribution: a scalar
of the wrong *type* is a `TypeError` at construction; an out-of-domain *value* is not checked here at
all, because a column-valued parameter could not be, and the two spellings must behave alike.

Two exceptions, both about a value the wire cannot carry rather than a domain: `coerce_n` refuses a
Python `int` outside `[0, 2**63 - 1]` and `coerce_int` one outside `Int64`, since a scalar count rides
the fast path as a kwarg. Those are bespoke and live at the bottom of this file.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import polars as pl
import pytest

from polars_stats import Binomial, DiscreteUniform, LogNormal, Normal
from tests._registry import ALL_SPECS, density

if TYPE_CHECKING:
    from tests._registry import DistSpec, InvalidCase, Param

# Values `_coerce` refuses whatever the parameter: not a number, or a `bool` (which subclasses `int`
# and would otherwise slip through as `0` / `1`).
_ALWAYS_REFUSED: list[object] = [None, True, False, [0.0], (0.0,), {"x": 0.0}]

_PARAMETERS = [
    pytest.param(spec, param, id=f"{spec.name}.{param.name}") for spec in ALL_SPECS for param in spec.parameters
]


@pytest.mark.parametrize(("spec", "param"), _PARAMETERS)
def test_a_wrong_scalar_type_is_refused(spec: DistSpec, param: Param) -> None:
    """Every parameter names itself and the type it wants, and rejects the other side's scalar.

    An integer parameter takes an `int` and refuses a `float`; every other parameter is the mirror
    image. Getting that backwards for one parameter of one distribution is the mistake this catches,
    and it is invisible to any test that only passes valid values.
    """
    wrong_number = 1.5 if param.integer else 2
    label = "an int" if param.integer else "a float"
    valid = {p.name: p.cast(v) for p, v in spec.pairs()}

    for bad in [*_ALWAYS_REFUSED, wrong_number]:
        with pytest.raises(TypeError, match=f"{param.name} should be {label} or IntoExprColumn"):
            spec.cls(**{**valid, param.name: bad})


@pytest.mark.parametrize(("spec", "param"), _PARAMETERS)
def test_every_into_expr_column_spelling_is_accepted(spec: DistSpec, param: Param) -> None:
    """`pl.Expr`, a column-name `str` and a `pl.Series` all construct, none of them checked here."""
    valid = {p.name: p.cast(v) for p, v in spec.pairs()}
    series = pl.Series(param.column, [param.cast(dict(spec.pairs())[param])], dtype=param.dtype)

    for spelling in (pl.col(param.column), param.column, series):
        spec.cls(**{**valid, param.name: spelling})  # must not raise


_INVALID = [pytest.param(spec, case, id=f"{spec.name}.{case.label}") for spec in ALL_SPECS for case in spec.invalid]


@pytest.mark.parametrize(("spec", "case"), _INVALID)
def test_an_invalid_scalar_defers_to_evaluation(spec: DistSpec, case: InvalidCase) -> None:
    """Construction succeeds on an out-of-domain scalar; the refusal is a `ComputeError` at evaluation.

    No early Python validation, deliberately: a column carrying the same value cannot be checked at
    construction, and the two spellings have to agree on where the error comes from.
    """
    dist = spec.build("scalar", case.params)  # must not raise
    with pytest.raises(pl.exceptions.ComputeError, match=case.match):
        pl.DataFrame({"x": [0.5]}).select(r=density(dist, pl.col("x")))


def test_column_parameters_defer_validation(spec: DistSpec) -> None:
    """A column's values are not knowable at construction, so neither spelling may raise there."""
    spec.build("column")  # must not raise
    spec.build("name")  # must not raise


@pytest.mark.parametrize(("cls", "median"), [(Normal, 0.0), (LogNormal, 1.0)], ids=["Normal", "LogNormal"])
def test_the_location_scale_families_default_to_the_standard_member(
    cls: type[Normal | LogNormal], median: float
) -> None:
    """`mu = 0`, `sigma = 1` with no arguments, handled by the same path as any other parameterisation.

    Probed at the standard member's own median, where the answer is exactly `0.5` and a defaulted
    `sigma` of anything but `1` still shows: a shifted `mu` moves the median off the point entirely.

    The only two distributions with defaults; every other parameter is required.
    """
    standard_std = 1.0 if cls is Normal else math.sqrt((math.e - 1) * math.e)

    got = pl.DataFrame({"x": [median]}).select(r=cls().cdf(pl.col("x")))["r"].item()
    assert got == pytest.approx(0.5, abs=1e-12)
    assert pl.DataFrame({"_": [0]}).select(r=cls().std()).item() == pytest.approx(standard_std)


# `n` and the discrete uniform's bounds are the parameters whose *value* is judged at construction: a
# Python `int` is turned into a `UInt64` / `Int64` literal and rides the fast paths as a kwarg, so a
# count outside the wire's range cannot be reported from the plugin the way an out-of-range `p` is.


@pytest.mark.parametrize(("bad_n", "match"), [(-1, "non-negative integer, got -1"), (2**63, "must be at most")])
def test_a_scalar_trial_count_outside_the_wire_range_raises_at_construction(bad_n: int, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Binomial(n=bad_n, p=0.5)


def test_a_trial_count_column_keeps_the_whole_u64_range() -> None:
    """The bound is the kwargs wire's, not the distribution's: a column keeps the whole `u64` range."""
    got = pl.DataFrame({"n": [2**64 - 1]}, schema={"n": pl.UInt64}).select(r=Binomial(pl.col("n"), 0.5).mean())
    assert got["r"][0] == pytest.approx((2**64 - 1) * 0.5)


@pytest.mark.parametrize(("name", "bound"), [("max", 2**63), ("min", -(2**63) - 1)])
def test_a_discrete_uniform_bound_outside_int64_raises_at_construction(name: str, bound: int) -> None:
    """A bound polars cannot hold as an `Int64` literal is refused when the literal is built."""
    kwargs: dict[str, int] = {"min": 0, "max": 5}
    kwargs[name] = bound
    with pytest.raises(ValueError, match=f"{name} must be in "):
        DiscreteUniform(**kwargs)


def test_signed_and_extreme_discrete_uniform_bounds_construct() -> None:
    """Signed bounds are ordinary here; validity is judged at evaluation."""
    DiscreteUniform(min=-10, max=-2)  # must not raise
    DiscreteUniform(min=-(2**62), max=2**62)  # must not raise
    DiscreteUniform(min=-(2**63), max=2**63 - 1)  # must not raise
