"""Null-bound contract: every method, both bounds, on and off the support.

An answer survives a null bound exactly when the other, known bound decides it. That is a rule about
bounds rather than about methods, so the table below is keyed on `(bounds, value)`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import polars as pl
import pytest

from polars_stats import Uniform

if TYPE_CHECKING:
    from collections.abc import Callable

# Every method must propagate a null bound to a null result. The value-keyed methods are evaluated at
# an on-support point, where both bounds enter the formula, so the null flows through whichever one is
# missing. The points where only one bound decides the answer are covered below.
_METHODS: dict[str, Callable[[Uniform], pl.Expr]] = {
    "pdf": lambda u: u.pdf(pl.lit(0.5)),
    "log_pdf": lambda u: u.log_pdf(pl.lit(0.5)),
    "cdf": lambda u: u.cdf(pl.lit(0.5)),
    "log_cdf": lambda u: u.log_cdf(pl.lit(0.5)),
    "sf": lambda u: u.sf(pl.lit(0.5)),
    "log_sf": lambda u: u.log_sf(pl.lit(0.5)),
    "ppf": lambda u: u.ppf(pl.lit(0.5)),
    "isf": lambda u: u.isf(pl.lit(0.5)),
    "mean": lambda u: u.mean(),
    "variance": lambda u: u.variance(),
    "std": lambda u: u.std(),
    "median": lambda u: u.median(),
    "entropy": lambda u: u.entropy(),
    "sample": lambda u: u.sample(seed=0),
    "samples": lambda u: u.samples(size=2, seed=0),
}

_SCHEMA = {"lo": pl.Float64, "hi": pl.Float64}


def _column_bounds() -> Uniform:
    return Uniform(min=pl.col("lo"), max=pl.col("hi"))


@pytest.mark.parametrize("expr_fn", _METHODS.values(), ids=list(_METHODS))
def test_method_propagates_null_in_min(expr_fn: Callable[[Uniform], pl.Expr]) -> None:
    df = pl.DataFrame({"lo": [0.0, None, 0.0], "hi": [1.0, 1.0, 1.0]}, schema=_SCHEMA)
    result = df.select(r=expr_fn(_column_bounds()))["r"]
    assert result.is_null().to_list() == [False, True, False]


@pytest.mark.parametrize("expr_fn", _METHODS.values(), ids=list(_METHODS))
def test_method_propagates_null_in_max(expr_fn: Callable[[Uniform], pl.Expr]) -> None:
    # `max` is the *second* plugin input, so a guard written against the first only would pass the
    # test above and fail here.
    df = pl.DataFrame({"lo": [0.0, 0.0, 0.0], "hi": [1.0, None, 1.0]}, schema=_SCHEMA)
    result = df.select(r=expr_fn(_column_bounds()))["r"]
    assert result.is_null().to_list() == [False, True, False]


_VALUE_METHODS = ("pdf", "log_pdf", "cdf", "log_cdf", "sf", "log_sf")

_BELOW_MIN: dict[str, float | None] = {
    "pdf": 0.0,
    "log_pdf": -math.inf,
    "cdf": 0.0,
    "log_cdf": -math.inf,
    "sf": 1.0,
    "log_sf": 0.0,
}
"""What the six value-keyed methods answer strictly below `min`, for every admissible `max`."""

_ABOVE_MAX: dict[str, float | None] = {
    "pdf": 0.0,
    "log_pdf": -math.inf,
    "cdf": 1.0,
    "log_cdf": 0.0,
    "sf": 0.0,
    "log_sf": -math.inf,
}
"""What they answer strictly above `max`, for every admissible `min`."""

_NULL: dict[str, float | None] = dict.fromkeys(_VALUE_METHODS)
"""Nothing survives: the null bound is the one that would have placed the point."""

# The `value == 1.0` row is where the two support conventions diverge: the densities treat the
# support as the closed `[min, max]`, so a point at `max` is inside it and still needs `min`, while
# `cdf` / `sf` and their logs are already saturated from `max` up. There is no lower-bound
# counterpart: every method leaves `min` itself to the interior, so `value == min` under a null
# `max` nulls throughout.
_NULL_BOUND_CASES: tuple[tuple[tuple[float | None, float | None], float, dict[str, float | None]], ...] = (
    ((None, 1.0), -5.0, _NULL),
    ((None, 1.0), 0.5, _NULL),
    ((None, 1.0), 1.0, {**_ABOVE_MAX, "pdf": None, "log_pdf": None}),
    ((None, 1.0), 5.0, _ABOVE_MAX),
    ((0.0, None), -5.0, _BELOW_MIN),
    ((0.0, None), 0.0, _NULL),
    ((0.0, None), 0.5, _NULL),
    ((0.0, None), 5.0, _NULL),
    ((None, None), -5.0, _NULL),
    ((None, None), 0.5, _NULL),
    ((None, None), 5.0, _NULL),
)


@pytest.mark.parametrize(
    ("bounds", "value", "expected"),
    _NULL_BOUND_CASES,
    ids=[f"min={lo},max={hi},v={v}" for (lo, hi), v, _ in _NULL_BOUND_CASES],
)
def test_value_keyed_answer_survives_only_when_the_known_bound_decides_it(
    bounds: tuple[float | None, float | None], value: float, expected: dict[str, float | None]
) -> None:
    # Row 0 carries known bounds so the null bound shares a chunk with a fully specified one: a
    # driver that answered per chunk rather than per row would still pass a one-row frame.
    lo, hi = bounds
    df = pl.DataFrame({"lo": [0.0, lo], "hi": [1.0, hi]}, schema=_SCHEMA)
    dist = _column_bounds()
    got = df.select(**{name: getattr(dist, name)(pl.lit(value)) for name in _VALUE_METHODS})
    assert {name: got[name][1] for name in _VALUE_METHODS} == expected
    assert all(got[name][0] is not None for name in _VALUE_METHODS), "the known-bounds row nulled"


@pytest.mark.parametrize("method", ["ppf", "isf"])
@pytest.mark.parametrize("quantile", [0.0, 0.25, 0.5, 0.75, 1.0, -0.5, 1.5])
@pytest.mark.parametrize("bounds", [(None, 1.0), (0.0, None), (None, None)], ids=str)
def test_inverse_nulls_under_a_null_bound(
    method: str, quantile: float, bounds: tuple[float | None, float | None]
) -> None:
    """Endpoints and out-of-range quantiles as well as interior ones: neither inverse has a
    bound-free branch anywhere in its domain, under either null bound."""
    lo, hi = bounds
    df = pl.DataFrame({"lo": [0.0, lo], "hi": [1.0, hi]}, schema=_SCHEMA)
    result = df.select(r=getattr(_column_bounds(), method)(quantile))["r"]
    assert result[1] is None
    assert (result[0] is not None) == (0.0 <= quantile <= 1.0), "the known-bounds row disagreed"


@pytest.mark.parametrize("hook", ["_pdf", "_log_pdf", "_cdf", "_log_cdf", "_sf", "_log_sf", "_ppf", "_isf"])
@pytest.mark.parametrize("bounds", [(None, 1.0), (0.0, None), (None, None)], ids=str)
def test_nan_value_stays_nan_under_a_null_bound(hook: str, bounds: tuple[float | None, float | None]) -> None:
    """A `NaN` evaluation point short-circuits before the branches, so a null bound does not null it.

    Reached through the private hook, since the public wrapper answers `NaN` on its own. Without the
    short-circuit `NaN` would be placed by whichever bound is known and answer that bound's constant.
    """
    lo, hi = bounds
    df = pl.DataFrame({"lo": [0.0, lo], "hi": [1.0, hi]}, schema=_SCHEMA)
    result = df.select(r=getattr(_column_bounds(), hook)(pl.lit(math.nan)))["r"].to_list()
    assert all(v is not None and math.isnan(v) for v in result), "a NaN evaluation point nulled"
