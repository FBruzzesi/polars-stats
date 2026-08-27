"""`p = 1`: the whole mass on `k = 1`, and the branch ordering that keeps `NaN` out of it.

At `p = 1` the shared entry `log1p(-p)` is `-inf`, so `floor(value) * log1p(-p)` is `NaN` for every
`value` in `[0, 1)` and `-inf` above it. Each of `_cdf`, `_log_cdf`, `_sf` and `_log_sf` answers its
below-support constant *before* forming that product, and each inverse short-circuits on `p == 1`
before dividing `-inf` by `-inf`. Nothing else pins that ordering: the scipy parity grid excludes
`p = 1` (scipy's generic discrete `ppf`, `isf`, `median` and `entropy` do not answer there) and the
per-method grids stop at `p = 0.8`, so a reordered `when` chain would leak `NaN` and still ship green.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import polars as pl
import pytest

from polars_stats import Geometric

if TYPE_CHECKING:
    from collections.abc import Callable

_NEG_INF = float("-inf")

# One row per (method, value) with the exact answer of the `p = 1` point mass.
_VALUE_KEYED: dict[str, tuple[Callable[[Geometric, pl.Expr], pl.Expr], list[tuple[float, float]]]] = {
    "pmf": (lambda g, v: g.pmf(v), [(-1.0, 0.0), (0.0, 0.0), (0.5, 0.0), (1.0, 1.0), (2.0, 0.0), (10.0, 0.0)]),
    "log_pmf": (
        lambda g, v: g.log_pmf(v),
        [(-1.0, _NEG_INF), (0.0, _NEG_INF), (0.5, _NEG_INF), (1.0, 0.0), (2.0, _NEG_INF)],
    ),
    "cdf": (lambda g, v: g.cdf(v), [(-1.0, 0.0), (0.0, 0.0), (0.5, 0.0), (1.0, 1.0), (2.0, 1.0), (1e6, 1.0)]),
    "log_cdf": (
        lambda g, v: g.log_cdf(v),
        [(-1.0, _NEG_INF), (0.0, _NEG_INF), (0.5, _NEG_INF), (1.0, 0.0), (2.0, 0.0), (1e6, 0.0)],
    ),
    "sf": (lambda g, v: g.sf(v), [(-1.0, 1.0), (0.0, 1.0), (0.5, 1.0), (1.0, 0.0), (2.0, 0.0), (1e6, 0.0)]),
    "log_sf": (
        lambda g, v: g.log_sf(v),
        [(-1.0, 0.0), (0.0, 0.0), (0.5, 0.0), (1.0, _NEG_INF), (2.0, _NEG_INF), (1e6, _NEG_INF)],
    ),
}


@pytest.mark.parametrize(("expr_fn", "cases"), _VALUE_KEYED.values(), ids=list(_VALUE_KEYED))
def test_value_keyed_method_at_p_one(
    expr_fn: Callable[[Geometric, pl.Expr], pl.Expr],
    cases: list[tuple[float, float]],
    unit_frame: pl.DataFrame,
) -> None:
    """Exact, and never `NaN`: `0 * -inf` must not reach the result on either side of the support."""
    for value, expected in cases:
        got = unit_frame.select(r=expr_fn(Geometric(p=1.0), pl.lit(value)))["r"].item()
        assert not math.isnan(got), f"{value} -> NaN"
        assert got == expected, f"{value} -> {got}, want {expected}"


@pytest.mark.parametrize(("expr_fn", "cases"), _VALUE_KEYED.values(), ids=list(_VALUE_KEYED))
def test_value_keyed_method_at_p_one_from_a_column(
    expr_fn: Callable[[Geometric, pl.Expr], pl.Expr],
    cases: list[tuple[float, float]],
) -> None:
    """The per-row path answers identically: `p = 1` is a value, not a constant-folding artefact."""
    frame = pl.DataFrame({"p": [1.0] * len(cases), "v": [v for v, _ in cases]})
    got = frame.select(r=expr_fn(Geometric(p=pl.col("p")), pl.col("v")))["r"].to_list()
    assert got == [e for _, e in cases]


@pytest.mark.parametrize("quantile", [0.0, 1e-300, 0.25, 0.5, 0.75, 1.0])
def test_both_inverses_collapse_to_the_mass_point(quantile: float, unit_frame: pl.DataFrame) -> None:
    """Every quantile inverts to `k = 1`, including the endpoints where the ratio would be `NaN`."""
    frame = unit_frame.select(ppf=Geometric(p=1.0).ppf(quantile), isf=Geometric(p=1.0).isf(quantile))
    assert frame.item(0, "ppf") == 1.0
    assert frame.item(0, "isf") == 1.0


def test_moments_at_p_one(unit_frame: pl.DataFrame) -> None:
    """A point mass at `k = 1`: mean `1`, no spread, and entropy `0` by the `0 log 0 = 0` convention."""
    g = Geometric(p=1.0)
    got = unit_frame.select(mean=g.mean(), variance=g.variance(), std=g.std(), median=g.median(), entropy=g.entropy())
    assert got.item(0, "mean") == 1.0
    assert got.item(0, "variance") == 0.0
    assert got.item(0, "std") == 0.0
    assert got.item(0, "median") == 1.0
    assert got.item(0, "entropy") == 0.0


def test_sample_at_p_one_is_always_the_mass_point() -> None:
    """Every trial succeeds, so every draw is `1`."""
    frame = pl.DataFrame({"_": range(64)})
    drawn = frame.select(r=Geometric(p=1.0).sample(seed=7))["r"]
    assert drawn.dtype == pl.UInt64
    assert drawn.unique().to_list() == [1]
