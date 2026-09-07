"""Null-bound contract: every method, both bounds, on and off the support.

A null bound nulls the answer wherever the point sits, so the `(bounds, value)` table below is
keyed to reach every region a known bound could have decided from, not to vary the expectation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from polars_stats import Uniform

if TYPE_CHECKING:
    from collections.abc import Callable

# Every method must propagate a null bound to a null result, evaluated here at an on-support point.
# The points outside the support, which the known bound alone places, are covered below.
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

_NULL_BOUND_CASES: tuple[tuple[tuple[float | None, float | None], float], ...] = (
    ((None, 1.0), -5.0),
    ((None, 1.0), 0.5),
    ((None, 1.0), 1.0),
    ((None, 1.0), 5.0),
    ((0.0, None), -5.0),
    ((0.0, None), 0.0),
    ((0.0, None), 0.5),
    ((0.0, None), 5.0),
    ((None, None), -5.0),
    ((None, None), 0.5),
    ((None, None), 5.0),
)


@pytest.mark.parametrize(
    ("bounds", "value"),
    _NULL_BOUND_CASES,
    ids=[f"min={lo},max={hi},v={v}" for (lo, hi), v in _NULL_BOUND_CASES],
)
def test_value_keyed_answer_nulls_under_a_null_bound(bounds: tuple[float | None, float | None], value: float) -> None:
    # Row 0 carries known bounds so the null bound shares a chunk with a fully specified one: a
    # driver that answered per chunk rather than per row would still pass a one-row frame.
    lo, hi = bounds
    df = pl.DataFrame({"lo": [0.0, lo], "hi": [1.0, hi]}, schema=_SCHEMA)
    dist = _column_bounds()
    got = df.select(**{name: getattr(dist, name)(pl.lit(value)) for name in _VALUE_METHODS})
    assert {name: got[name][1] for name in _VALUE_METHODS} == dict.fromkeys(_VALUE_METHODS)
    assert all(got[name][0] is not None for name in _VALUE_METHODS), "the known-bounds row nulled"


@pytest.mark.parametrize("method", ["ppf", "isf"])
@pytest.mark.parametrize("quantile", [0.0, 0.25, 0.5, 0.75, 1.0, -0.5, 1.5])
@pytest.mark.parametrize("bounds", [(None, 1.0), (0.0, None), (None, None)], ids=str)
def test_inverse_nulls_under_a_null_bound(
    method: str, quantile: float, bounds: tuple[float | None, float | None]
) -> None:
    """Endpoints and out-of-range quantiles as well as interior ones, under either null bound."""
    lo, hi = bounds
    df = pl.DataFrame({"lo": [0.0, lo], "hi": [1.0, hi]}, schema=_SCHEMA)
    result = df.select(r=getattr(_column_bounds(), method)(quantile))["r"]
    assert result[1] is None
    assert (result[0] is not None) == (0.0 <= quantile <= 1.0), "the known-bounds row disagreed"
