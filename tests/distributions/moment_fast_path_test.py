"""Validation contract of the constant-parameter moment fast path.

Every moment validates its parameters through a Rust plugin (`normal_sigma`, `uniform_range`,
`bernoulli_proba`, ..., or the `beta_entropy` / `binomial_entropy` formula itself). With all-scalar
parameters that plugin runs once on length-1 `pl.lit` inputs instead of per row.
`tests/property/moment_test.py` pins bit-equality against the per-row path for *valid* parameters; this
module pins what that test does not reach:

* both paths raise the same `ComputeError` on an invalid scalar parameterisation, a non-finite
  parameter included (finiteness is the library's own check, whether or not `statrs` accepts the value);
* the validator runs whatever the frame height, so an empty frame still raises on the scalar path
  and yields one row on a valid one, while the per-row path returns empty.

A Python `None` parameter and a negative scalar `n` are rejected at construction (`coerce_param`,
`coerce_n`), so neither has a case here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from polars_stats import (
    Bernoulli,
    Beta,
    Binomial,
    DiscreteUniform,
    Geometric,
    LogNormal,
    Normal,
    Uniform,
)
from tests._polars_compat import assert_series_equal

if TYPE_CHECKING:
    from polars_stats.distributions._base import _UnivariateDistribution

_NAN = float("nan")
_INF = float("inf")


def _col(value: float, dtype: pl.DataType | None = None) -> pl.Expr:
    """A full-length constant column (`pl.repeat`), forcing the general per-row path for a scalar value."""
    return pl.repeat(value, n=pl.len(), dtype=dtype)


# id -> (scalar instance, equivalent per-row instance). `variance` is the representative
# moment: it routes through the parameter validator for every distribution (Normal/LogNormal/Binomial
# via `_moment`, Uniform via `range`). The `inf` cases guard that the fast
# path applies the library's own finiteness check where `statrs` alone would accept the value.
_CASES: dict[str, tuple[_UnivariateDistribution, _UnivariateDistribution]] = {
    "normal mu=nan": (Normal(_NAN, 1.0), Normal(_col(_NAN), _col(1.0))),
    "normal mu=inf": (Normal(_INF, 1.0), Normal(_col(_INF), _col(1.0))),
    "normal std=0": (Normal(0.0, 0.0), Normal(_col(0.0), _col(0.0))),
    "normal std=-1": (Normal(0.0, -1.0), Normal(_col(0.0), _col(-1.0))),
    "normal std=nan": (Normal(0.0, _NAN), Normal(_col(0.0), _col(_NAN))),
    "normal std=inf": (Normal(0.0, _INF), Normal(_col(0.0), _col(_INF))),
    "lognormal sigma=-1": (LogNormal(0.0, -1.0), LogNormal(_col(0.0), _col(-1.0))),
    "lognormal sigma=nan": (LogNormal(0.0, _NAN), LogNormal(_col(0.0), _col(_NAN))),
    "lognormal sigma=inf": (LogNormal(0.0, _INF), LogNormal(_col(0.0), _col(_INF))),
    "lognormal mu=inf": (LogNormal(_INF, 1.0), LogNormal(_col(_INF), _col(1.0))),
    "uniform max<min": (Uniform(2.0, 1.0), Uniform(_col(2.0), _col(1.0))),
    "uniform max=min": (Uniform(1.0, 1.0), Uniform(_col(1.0), _col(1.0))),
    "uniform min=nan": (Uniform(_NAN, 1.0), Uniform(_col(_NAN), _col(1.0))),
    "bernoulli p=1.5": (Bernoulli(1.5), Bernoulli(_col(1.5))),
    "bernoulli p=-0.1": (Bernoulli(-0.1), Bernoulli(_col(-0.1))),
    "bernoulli p=nan": (Bernoulli(_NAN), Bernoulli(_col(_NAN))),
    # The geometric support excludes the `p = 0` point mass its Bernoulli counterpart accepts.
    "geometric p=0": (Geometric(0.0), Geometric(_col(0.0))),
    "geometric p=-0.1": (Geometric(-0.1), Geometric(_col(-0.1))),
    "geometric p=1.5": (Geometric(1.5), Geometric(_col(1.5))),
    "geometric p=nan": (Geometric(_NAN), Geometric(_col(_NAN))),
    # Inverted bounds: the one invalid parameterisation the discrete uniform has.
    "discreteuniform min>max": (DiscreteUniform(6, 1), DiscreteUniform(_col(6), _col(1))),
    "binomial p=1.5": (Binomial(5, 1.5), Binomial(_col(5, pl.Int64()), _col(1.5))),
    "binomial p=nan": (Binomial(5, _NAN), Binomial(_col(5, pl.Int64()), _col(_NAN))),
    "beta a=0": (Beta(0.0, 1.0), Beta(_col(0.0), _col(1.0))),
    "beta b=-1": (Beta(2.0, -1.0), Beta(_col(2.0), _col(-1.0))),
    "beta a=nan": (Beta(_NAN, 1.0), Beta(_col(_NAN), _col(1.0))),
    "beta a=inf": (Beta(_INF, 1.0), Beta(_col(_INF), _col(1.0))),
}


@pytest.mark.parametrize(("scalar", "per_row"), _CASES.values(), ids=list(_CASES))
def test_scalar_and_column_paths_agree_on_validation(
    scalar: _UnivariateDistribution, per_row: _UnivariateDistribution
) -> None:
    """The moment fast path and the per-row path agree on what is a valid parameterisation."""
    frame = pl.DataFrame({"_": range(4)})

    with pytest.raises(pl.exceptions.ComputeError):
        frame.select(r=scalar.variance())
    with pytest.raises(pl.exceptions.ComputeError):
        frame.select(r=per_row.variance())


@pytest.mark.parametrize(
    ("scalar", "per_row"),
    [(Binomial(5, 1.5), Binomial(_col(5, pl.Int64()), _col(1.5)))],
    ids=["p=1.5"],
)
def test_binomial_entropy_scalar_and_column_paths_agree_on_validation(scalar: Binomial, per_row: Binomial) -> None:
    """`Binomial.entropy` is its own plugin (the Rust support sum), so its validation is pinned on its own."""
    frame = pl.DataFrame({"_": range(4)})
    with pytest.raises(pl.exceptions.ComputeError):
        frame.select(r=scalar.entropy())
    with pytest.raises(pl.exceptions.ComputeError):
        frame.select(r=per_row.entropy())


def test_moment_fast_path_on_empty_frame_is_a_scalar() -> None:
    """On a zero-row frame the scalar moment is still one row, and still validates.

    Both follow from the scalar path being built from length-1 literals, exactly as
    `df.head(0).select(pl.lit(1.0))` is one row. The per-row path's column pass sees no value on an
    empty frame, so it returns empty without raising.
    """
    empty = pl.DataFrame({"_": []}, schema={"_": pl.Int64})

    valid_fast = empty.select(r=Normal(0.0, 2.0).variance())
    valid_slow = empty.select(r=Normal(_col(0.0), _col(2.0)).variance())
    assert valid_fast.height == 1
    assert valid_fast["r"].to_list() == [4.0]
    assert valid_slow.height == 0

    with pytest.raises(pl.exceptions.ComputeError):
        empty.select(r=Normal(0.0, -1.0).variance())
    assert empty.select(r=Normal(_col(0.0), _col(-1.0)).variance()).height == 0


# id -> (scalar instance, equivalent per-row instance, expected constant moment value). Degenerate
# but *valid* parameterisations where a moment collapses to a closed value: the discrete entropies
# use the `0 log 0 = 0` convention at the mass-collapsing endpoints, and `n = 0` is a valid (point
# mass at 0) binomial. The fast path must reach the same value as the per-row path, not raise.
_DEGENERATE: dict[str, tuple[_UnivariateDistribution, _UnivariateDistribution, float]] = {
    "bernoulli p=0 entropy": (Bernoulli(0.0), Bernoulli(_col(0.0)), 0.0),
    "bernoulli p=1 entropy": (Bernoulli(1.0), Bernoulli(_col(1.0)), 0.0),
    "geometric p=1 entropy": (Geometric(1.0), Geometric(_col(1.0)), 0.0),
    "binomial p=0 entropy": (Binomial(5, 0.0), Binomial(_col(5, pl.Int64()), _col(0.0)), 0.0),
    "binomial p=1 entropy": (Binomial(5, 1.0), Binomial(_col(5, pl.Int64()), _col(1.0)), 0.0),
    "binomial n=0 entropy": (Binomial(0, 0.5), Binomial(_col(0, pl.Int64()), _col(0.5)), 0.0),
    # `min == max` is the legitimate one-point mass: entropy collapses to 0.
    "discreteuniform min=max entropy": (
        DiscreteUniform(4, 4),
        DiscreteUniform(_col(4, pl.Int64()), _col(4, pl.Int64())),
        0.0,
    ),
}


@pytest.mark.parametrize(("scalar", "per_row", "expected"), _DEGENERATE.values(), ids=list(_DEGENERATE))
def test_degenerate_valid_params_entropy(
    scalar: _UnivariateDistribution, per_row: _UnivariateDistribution, expected: float
) -> None:
    """Mass-collapsing endpoints (`p in {0, 1}`, `n = 0`) give a finite entropy on both paths."""
    frame = pl.DataFrame({"_": range(4)})

    assert frame.select(r=scalar.entropy())["r"].to_list() == [expected]
    both = frame.select(fast=scalar.entropy(), slow=per_row.entropy())
    assert_series_equal(both["fast"], both["slow"], check_names=False, check_exact=True)
    assert both["slow"].to_list() == [expected] * 4
