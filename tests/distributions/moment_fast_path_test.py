"""Validation contract of the constant-parameter moment fast path (validate-once).

The closed-form moments (`mean` / `variance` / `std` / `entropy`) and the closed forms of `Uniform`
/ `Bernoulli` route their *validation* through a small Rust plugin (`normal_sigma`, `uniform_range`,
`bernoulli_proba`, `binomial_params`, `lognormal_sigma`, `beta_params`, plus the `binomial_entropy` /
`beta_entropy` parameter-keyed formulas). For
all-scalar parameters that plugin is called once on length-1 `pl.lit` inputs instead of per row.
`tests/property/moment_test.py` pins bit-equality against the per-row path for *valid* parameters;
this module pins the parts that test does not reach:

* **Both paths agree on validation.** An invalid scalar parameterisation must raise the same
  `ComputeError` as the equivalent per-row columns; an accepted non-finite one (a positive-infinite
  scale, which `statrs` allows) must produce identical output on both. The fast path cannot quietly
  accept or reject something the per-row path does not. Unlike the value-keyed fast-path test, this
  covers `Uniform` and `Bernoulli` too: their moments *do* route through a validator, so the scalar
  path can drift there.
* **The validator runs once but only when there is work.** Because the scalar validator rides as a
  *gated* length-1 plugin input (not as kwargs), an empty frame skips it entirely: both paths return
  an empty column and neither raises. This differs from the value-keyed fast path, whose kwargs
  validation raises even on a zero-row frame (`value_keyed_fast_path_test.py`); it is pinned here so
  the asymmetry is intentional and cannot regress silently.

A Python `None` parameter cannot reach either path (`coerce_param` rejects it at construction), so
there is no null-scalar case here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from polars_stats import Bernoulli, Beta, Binomial, LogNormal, Normal, Uniform
from tests._polars_compat import assert_series_equal

if TYPE_CHECKING:
    from polars_stats.distributions._base import _UnivariateDistribution

_NAN = float("nan")
_INF = float("inf")


def _col(value: float, dtype: pl.DataType | None = None) -> pl.Expr:
    """A full-length constant column (`pl.repeat`), forcing the general per-row path for a scalar value."""
    return pl.repeat(value, n=pl.len(), dtype=dtype)


# id -> (scalar instance, equivalent per-row instance, should_raise). `variance` is the representative
# moment: it routes through the parameter validator for every distribution (Normal/LogNormal/Binomial
# via `_moment`, Uniform via `range`, Bernoulli via `_checked_p`). The accepted-`inf` cases guard that
# the fast path does not reject a parameterisation the per-row path accepts.
_CASES: dict[str, tuple[_UnivariateDistribution, _UnivariateDistribution, bool]] = {
    "normal mu=nan": (Normal(_NAN, 1.0), Normal(_col(_NAN), _col(1.0)), True),
    "normal std=0": (Normal(0.0, 0.0), Normal(_col(0.0), _col(0.0)), True),
    "normal std=-1": (Normal(0.0, -1.0), Normal(_col(0.0), _col(-1.0)), True),
    "normal std=nan": (Normal(0.0, _NAN), Normal(_col(0.0), _col(_NAN)), True),
    "normal std=inf (accepted)": (Normal(0.0, _INF), Normal(_col(0.0), _col(_INF)), False),
    "lognormal sigma=-1": (LogNormal(0.0, -1.0), LogNormal(_col(0.0), _col(-1.0)), True),
    "lognormal sigma=nan": (LogNormal(0.0, _NAN), LogNormal(_col(0.0), _col(_NAN)), True),
    "lognormal sigma=inf (accepted)": (LogNormal(0.0, _INF), LogNormal(_col(0.0), _col(_INF)), False),
    "uniform max<min": (Uniform(2.0, 1.0), Uniform(_col(2.0), _col(1.0)), True),
    "uniform max=min": (Uniform(1.0, 1.0), Uniform(_col(1.0), _col(1.0)), True),
    "uniform min=nan": (Uniform(_NAN, 1.0), Uniform(_col(_NAN), _col(1.0)), True),
    "bernoulli p=1.5": (Bernoulli(1.5), Bernoulli(_col(1.5)), True),
    "bernoulli p=-0.1": (Bernoulli(-0.1), Bernoulli(_col(-0.1)), True),
    "bernoulli p=nan": (Bernoulli(_NAN), Bernoulli(_col(_NAN)), True),
    "binomial n=-1": (Binomial(-1, 0.5), Binomial(_col(-1, pl.Int64()), _col(0.5)), True),
    "binomial p=1.5": (Binomial(5, 1.5), Binomial(_col(5, pl.Int64()), _col(1.5)), True),
    "binomial p=nan": (Binomial(5, _NAN), Binomial(_col(5, pl.Int64()), _col(_NAN)), True),
    "beta a=0": (Beta(0.0, 1.0), Beta(_col(0.0), _col(1.0)), True),
    "beta b=-1": (Beta(2.0, -1.0), Beta(_col(2.0), _col(-1.0)), True),
    "beta a=nan": (Beta(_NAN, 1.0), Beta(_col(_NAN), _col(1.0)), True),
    # An infinite shape is *rejected* by statrs, unlike the Normal / LogNormal scale.
    "beta a=inf (rejected)": (Beta(_INF, 1.0), Beta(_col(_INF), _col(1.0)), True),
}


@pytest.mark.parametrize(("scalar", "per_row", "should_raise"), _CASES.values(), ids=list(_CASES))
def test_scalar_and_column_paths_agree_on_validation(
    scalar: _UnivariateDistribution, per_row: _UnivariateDistribution, *, should_raise: bool
) -> None:
    """The moment fast path and the per-row path agree on what is a valid parameterisation."""
    frame = pl.DataFrame({"_": range(4)})

    if should_raise:
        with pytest.raises(pl.exceptions.ComputeError):
            frame.select(r=scalar.variance())
        with pytest.raises(pl.exceptions.ComputeError):
            frame.select(r=per_row.variance())
    else:
        fast = frame.select(r=scalar.variance())["r"]
        slow = frame.select(r=per_row.variance())["r"]
        assert_series_equal(fast, slow, check_exact=True)


@pytest.mark.parametrize(
    ("scalar", "per_row"),
    [
        (Binomial(-1, 0.5), Binomial(_col(-1, pl.Int64()), _col(0.5))),
        (Binomial(5, 1.5), Binomial(_col(5, pl.Int64()), _col(1.5))),
    ],
    ids=["n=-1", "p=1.5"],
)
def test_binomial_entropy_scalar_and_column_paths_agree_on_validation(scalar: Binomial, per_row: Binomial) -> None:
    """`Binomial.entropy` has its own scalar branch (the Rust support sum on length-1 inputs).

    It is gated by `_moment` like the closed-form moments, so an invalid `(n, p)` must raise on both
    paths there too, not only through `variance`.
    """
    frame = pl.DataFrame({"_": range(4)})
    with pytest.raises(pl.exceptions.ComputeError):
        frame.select(r=scalar.entropy())
    with pytest.raises(pl.exceptions.ComputeError):
        frame.select(r=per_row.entropy())


def test_moment_fast_path_on_empty_frame_matches_per_row() -> None:
    """On a zero-row frame, valid scalar params give an empty column matching the per-row path.

    The invalid-parameter case on an empty frame is deliberately *not* asserted. Unlike the
    value-keyed fast path (whose kwargs validation raises unconditionally, pinned by
    `value_keyed_fast_path_test.py`), the moment fast path validates via a *gated* length-1 plugin
    input, and whether polars evaluates that input when the gated output is empty is
    optimizer-dependent: some versions elide it and return empty, others run it and raise. Both are
    acceptable for an invalid parameterisation over zero rows, so it is left unspecified rather than
    pinned to one version's behaviour.
    """
    empty = pl.DataFrame({"_": []}, schema={"_": pl.Int64})

    valid_fast = empty.select(r=Normal(0.0, 2.0).variance())
    valid_slow = empty.select(r=Normal(_col(0.0), _col(2.0)).variance())
    assert valid_fast.height == 0
    assert_series_equal(valid_fast["r"], valid_slow["r"], check_exact=True)


# id -> (scalar instance, equivalent per-row instance, expected constant moment value). Degenerate
# but *valid* parameterisations where a moment collapses to a closed value: the discrete entropies
# use the `0 log 0 = 0` convention at the mass-collapsing endpoints, and `n = 0` is a valid (point
# mass at 0) binomial. The fast path must reach the same value as the per-row path, not raise.
_DEGENERATE: dict[str, tuple[_UnivariateDistribution, _UnivariateDistribution, float]] = {
    "bernoulli p=0 entropy": (Bernoulli(0.0), Bernoulli(_col(0.0)), 0.0),
    "bernoulli p=1 entropy": (Bernoulli(1.0), Bernoulli(_col(1.0)), 0.0),
    "binomial p=0 entropy": (Binomial(5, 0.0), Binomial(_col(5, pl.Int64()), _col(0.0)), 0.0),
    "binomial p=1 entropy": (Binomial(5, 1.0), Binomial(_col(5, pl.Int64()), _col(1.0)), 0.0),
    "binomial n=0 entropy": (Binomial(0, 0.5), Binomial(_col(0, pl.Int64()), _col(0.5)), 0.0),
}


@pytest.mark.parametrize(("scalar", "per_row", "expected"), _DEGENERATE.values(), ids=list(_DEGENERATE))
def test_degenerate_valid_params_entropy(
    scalar: _UnivariateDistribution, per_row: _UnivariateDistribution, expected: float
) -> None:
    """Mass-collapsing endpoints (`p in {0, 1}`, `n = 0`) give a finite entropy on both paths."""
    frame = pl.DataFrame({"_": range(4)})

    fast = frame.select(r=scalar.entropy())["r"]
    slow = frame.select(r=per_row.entropy())["r"]
    assert_series_equal(fast, slow, check_exact=True)
    assert fast.to_list() == [expected] * 4
