"""Validation contract of the constant-parameter value-keyed fast path.

All-scalar parameters route every value-keyed method through the ``<name>_<method>_scalar`` plugin, which
validates and builds the distribution once instead of per row. `tests/property/value_keyed_test.py` pins
bit-equality against the per-row path for *valid* parameters; this module pins what that test does not reach:

* both paths raise the same `ComputeError` on an invalid parameterisation, and agree on an accepted
  degenerate one (`Bernoulli`'s point masses, `Geometric`'s `p = 1`, `DiscreteUniform`'s one-point support).
  A non-finite parameter is invalid by the library's own check, whether or not `statrs` accepts it;
* constants validate up front, columns value by value. A Python scalar in kwargs and a length-1
  expression (`pl.lit`, an aggregate) are checked before any value is read, so an invalid one raises on a
  zero-row frame; an empty parameter column has no values to reject and returns empty. One constant beside
  a column still aligns, and a mismatched column is still reported.

A Python `None` parameter and a negative scalar `n` are rejected at construction (`coerce_param`, `coerce_n`),
so neither has a case here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from polars_stats import Bernoulli, Beta, Binomial, DiscreteUniform, Exponential, Geometric, LogNormal, Normal, Uniform
from polars_stats.distributions._base import ContinuousDistribution
from tests._polars_compat import assert_series_equal

if TYPE_CHECKING:
    from collections.abc import Callable

    from polars_stats.distributions._base import _UnivariateDistribution

_NAN = float("nan")
_INF = float("inf")


def _density(dist: _UnivariateDistribution, value: pl.Expr) -> pl.Expr:
    """`pdf` (continuous) or `pmf` (discrete), the representative value-keyed method per family."""
    return dist.pdf(value) if isinstance(dist, ContinuousDistribution) else dist.pmf(value)  # type: ignore[attr-defined]


def _col(value: float, dtype: pl.DataType | None = None) -> pl.Expr:
    """A full-length constant column (`pl.repeat`), forcing the general per-row path for a scalar value."""
    return pl.repeat(value, n=pl.len(), dtype=dtype)


# Each case builds the same parameterisation two ways -- all-scalar (constant-parameter fast path)
# and all-column (general per-row path) -- and records whether evaluating a value-keyed method
# should raise. `int` `n` for binomial keeps its `Int64` column dtype.
# id -> (scalar factory, column factory, should_raise)
_CASES: dict[str, tuple[Callable[[], _UnivariateDistribution], Callable[[], _UnivariateDistribution], bool]] = {
    "bernoulli p=nan": (lambda: Bernoulli(_NAN), lambda: Bernoulli(_col(_NAN)), True),
    "bernoulli p=-0.1": (lambda: Bernoulli(-0.1), lambda: Bernoulli(_col(-0.1)), True),
    "bernoulli p=1.5": (lambda: Bernoulli(1.5), lambda: Bernoulli(_col(1.5)), True),
    # The degenerate point masses are legitimate parameterisations; both paths must accept them.
    "bernoulli p=0 (accepted)": (lambda: Bernoulli(0.0), lambda: Bernoulli(_col(0.0)), False),
    "bernoulli p=1 (accepted)": (lambda: Bernoulli(1.0), lambda: Bernoulli(_col(1.0)), False),
    "normal mu=nan": (lambda: Normal(_NAN, 1.0), lambda: Normal(_col(_NAN), _col(1.0)), True),
    "normal mu=inf": (lambda: Normal(_INF, 1.0), lambda: Normal(_col(_INF), _col(1.0)), True),
    "normal std=0": (lambda: Normal(0.0, 0.0), lambda: Normal(_col(0.0), _col(0.0)), True),
    "normal std=-1": (lambda: Normal(0.0, -1.0), lambda: Normal(_col(0.0), _col(-1.0)), True),
    "normal std=nan": (lambda: Normal(0.0, _NAN), lambda: Normal(_col(0.0), _col(_NAN)), True),
    # `statrs` accepts a positive-infinite scale; the library's own finiteness check refuses it, and
    # both paths must apply that check.
    "normal std=inf": (lambda: Normal(0.0, _INF), lambda: Normal(_col(0.0), _col(_INF)), True),
    "lognormal sigma=nan": (lambda: LogNormal(0.0, _NAN), lambda: LogNormal(_col(0.0), _col(_NAN)), True),
    "lognormal sigma=-1": (lambda: LogNormal(0.0, -1.0), lambda: LogNormal(_col(0.0), _col(-1.0)), True),
    "lognormal sigma=inf": (lambda: LogNormal(0.0, _INF), lambda: LogNormal(_col(0.0), _col(_INF)), True),
    "lognormal mu=inf": (lambda: LogNormal(_INF, 1.0), lambda: LogNormal(_col(_INF), _col(1.0)), True),
    "binomial p=nan": (lambda: Binomial(5, _NAN), lambda: Binomial(_col(5, pl.Int64()), _col(_NAN)), True),
    "binomial p=1.5": (lambda: Binomial(5, 1.5), lambda: Binomial(_col(5, pl.Int64()), _col(1.5)), True),
    "beta a=nan": (lambda: Beta(_NAN, 1.0), lambda: Beta(_col(_NAN), _col(1.0)), True),
    "beta a=0": (lambda: Beta(0.0, 1.0), lambda: Beta(_col(0.0), _col(1.0)), True),
    "beta b=-1": (lambda: Beta(2.0, -1.0), lambda: Beta(_col(2.0), _col(-1.0)), True),
    "beta a=inf": (lambda: Beta(_INF, 1.0), lambda: Beta(_col(_INF), _col(1.0)), True),
    "exponential rate=nan": (lambda: Exponential(_NAN), lambda: Exponential(_col(_NAN)), True),
    "exponential rate=0": (lambda: Exponential(0.0), lambda: Exponential(_col(0.0)), True),
    "exponential rate=-1": (lambda: Exponential(-1.0), lambda: Exponential(_col(-1.0)), True),
    # `statrs` accepts an infinite rate as the degenerate point mass at 0; the library's finiteness
    # check refuses it, as for the Normal scale.
    "exponential rate=inf": (lambda: Exponential(_INF), lambda: Exponential(_col(_INF)), True),
    "geometric p=nan": (lambda: Geometric(_NAN), lambda: Geometric(_col(_NAN)), True),
    "geometric p=1.5": (lambda: Geometric(1.5), lambda: Geometric(_col(1.5)), True),
    # `p = 0` is the one endpoint Geometric rejects where Bernoulli accepts it: the trial count of a
    # never-succeeding trial is not representable.
    "geometric p=0": (lambda: Geometric(0.0), lambda: Geometric(_col(0.0)), True),
    "geometric p=1 (accepted)": (lambda: Geometric(1.0), lambda: Geometric(_col(1.0)), False),
    "discreteuniform min>max": (
        lambda: DiscreteUniform(6, 1),
        lambda: DiscreteUniform(_col(6, pl.Int64()), _col(1, pl.Int64())),
        True,
    ),
    # `min == max` is the legitimate one-point mass; both paths must accept it identically.
    "discreteuniform min=max (accepted)": (
        lambda: DiscreteUniform(3, 3),
        lambda: DiscreteUniform(_col(3, pl.Int64()), _col(3, pl.Int64())),
        False,
    ),
    "uniform max<min": (lambda: Uniform(5.0, 2.0), lambda: Uniform(_col(5.0), _col(2.0)), True),
    # `min == max` is an empty support for a *continuous* uniform, unlike the discrete one above, so
    # here the two distributions disagree and both uniform paths must reject it.
    "uniform max=min": (lambda: Uniform(3.0, 3.0), lambda: Uniform(_col(3.0), _col(3.0)), True),
    "uniform max=nan": (lambda: Uniform(0.0, _NAN), lambda: Uniform(_col(0.0), _col(_NAN)), True),
    # Individually finite bounds whose width overflows `f64`: rejected by uniform's own guard rather
    # than by statrs, and the fast path must apply it as the per-row path does.
    "uniform width overflows": (lambda: Uniform(-1e308, 1e308), lambda: Uniform(_col(-1e308), _col(1e308)), True),
}


@pytest.mark.parametrize(("scalar_mk", "column_mk", "should_raise"), _CASES.values(), ids=list(_CASES))
def test_scalar_and_column_paths_agree_on_validation(
    scalar_mk: Callable[[], _UnivariateDistribution],
    column_mk: Callable[[], _UnivariateDistribution],
    *,
    should_raise: bool,
) -> None:
    """The fast path and the per-row path agree on what is a valid parameterisation."""
    frame = pl.DataFrame({"x": [0.5, 1.0, 2.0]})
    scalar, column = scalar_mk(), column_mk()

    if should_raise:
        with pytest.raises(pl.exceptions.ComputeError):
            frame.select(r=_density(scalar, pl.col("x")))
        with pytest.raises(pl.exceptions.ComputeError):
            frame.select(r=_density(column, pl.col("x")))
    else:
        fast = frame.select(r=_density(scalar, pl.col("x")))["r"]
        per_row = frame.select(r=_density(column, pl.col("x")))["r"]
        assert_series_equal(fast, per_row, check_exact=True)


_INT = pl.Int64()

# One invalid parameterisation per Rust driver, spelled three ways: Python scalar, length-1 literal,
# full-length column. id -> (scalar, literal, column, message fragment)
_INVALID_PER_DRIVER: dict[
    str, tuple[_UnivariateDistribution, _UnivariateDistribution, _UnivariateDistribution, str]
] = {
    "exponential rate=-1": (Exponential(-1.0), Exponential(pl.lit(-1.0)), Exponential(_col(-1.0)), "rate must be"),
    "uniform max<min": (
        Uniform(5.0, 2.0),
        Uniform(pl.lit(5.0), pl.lit(2.0)),
        Uniform(_col(5.0), _col(2.0)),
        "max must be",
    ),
    "normal std=-1": (
        Normal(0.0, -1.0),
        Normal(pl.lit(0.0), pl.lit(-1.0)),
        Normal(_col(0.0), _col(-1.0)),
        "sigma must be",
    ),
    "discreteuniform min>max": (
        DiscreteUniform(6, 1),
        DiscreteUniform(pl.lit(6, dtype=_INT), pl.lit(1, dtype=_INT)),
        DiscreteUniform(_col(6, _INT), _col(1, _INT)),
        "max must be",
    ),
}


@pytest.mark.parametrize(
    ("scalar", "literal", "column", "fragment"), _INVALID_PER_DRIVER.values(), ids=list(_INVALID_PER_DRIVER)
)
def test_constant_parameters_validate_on_empty_input(
    scalar: _UnivariateDistribution, literal: _UnivariateDistribution, column: _UnivariateDistribution, fragment: str
) -> None:
    """An invalid constant raises with no rows to observe it; an invalid empty column returns empty."""
    empty = pl.DataFrame({"x": []}, schema={"x": pl.Float64})

    for constant in (scalar, literal):
        with pytest.raises(pl.exceptions.ComputeError, match=fragment):
            empty.select(r=_density(constant, pl.col("x")))

    assert empty.select(r=_density(column, pl.col("x"))).height == 0


def test_a_constant_the_cast_rejects_also_raises_on_an_empty_frame() -> None:
    """`n = -5` is refused by the strict `Int64 -> UInt64` cast, not a domain check, and still before any row.

    A negative Python-scalar `n` has no spelling here: `coerce_n` rejects it at construction.
    """
    empty = pl.DataFrame({"x": []}, schema={"x": pl.Float64})

    with pytest.raises(pl.exceptions.PolarsError, match="n must be a non-negative integer"):
        empty.select(r=Binomial(n=pl.lit(-5, dtype=_INT), p=pl.lit(0.5)).pmf(pl.col("x")))

    assert empty.select(r=Binomial(n=_col(-5, _INT), p=_col(0.5)).pmf(pl.col("x"))).height == 0


# One null length-1 parameterisation per Rust driver, plus Binomial, whose `n` has its own coercer.
_NULL_CONSTANT_PER_DRIVER: dict[str, _UnivariateDistribution] = {
    "exponential": Exponential(pl.lit(None, dtype=pl.Float64)),
    "uniform": Uniform(pl.lit(None, dtype=pl.Float64), pl.lit(1.0)),
    "normal": Normal(pl.lit(0.0), pl.lit(None, dtype=pl.Float64)),
    "binomial": Binomial(n=pl.lit(None, dtype=_INT), p=pl.lit(0.5)),
    "discreteuniform": DiscreteUniform(pl.lit(None, dtype=_INT), pl.lit(6, dtype=_INT)),
}


@pytest.mark.parametrize("dist", _NULL_CONSTANT_PER_DRIVER.values(), ids=list(_NULL_CONSTANT_PER_DRIVER))
def test_a_null_constant_parameter_nulls_every_row(dist: _UnivariateDistribution) -> None:
    """A null length-1 parameter nulls every row, as a null parameter column does."""
    frame = pl.DataFrame({"x": [0.5, 1.0, 2.0]})

    result = frame.select(r=_density(dist, pl.col("x")))["r"]

    assert result.len() == frame.height
    assert result.null_count() == frame.height


@pytest.mark.parametrize("dist", _NULL_CONSTANT_PER_DRIVER.values(), ids=list(_NULL_CONSTANT_PER_DRIVER))
def test_a_null_constant_parameter_still_gates_the_value_dtype(dist: _UnivariateDistribution) -> None:
    """A null length-1 parameter does not skip the value column's dtype gate: a `String` column is still refused."""
    strings = pl.DataFrame({"x": ["a", "b"]})

    with pytest.raises(pl.exceptions.ComputeError, match="must be a numeric column"):
        strings.select(r=_density(dist, pl.col("x")))


@pytest.mark.parametrize("dtype", [pl.Int64, pl.UInt64, pl.Int32])
def test_discreteuniform_integer_points_agree_across_bound_spellings(dtype: pl.DataType) -> None:
    """Integer points stay exact under every bound spelling; `UInt64` is the one dtype with its own arm."""
    frame = pl.DataFrame({"x": [0, 1, 3, 6, 7]}, schema={"x": dtype})
    expected = frame.select(r=DiscreteUniform(1, 6).pmf(pl.col("x")))["r"]

    literal = DiscreteUniform(pl.lit(1, dtype=_INT), pl.lit(6, dtype=_INT))
    column = DiscreteUniform(_col(1, _INT), _col(6, _INT))
    for dist in (literal, column):
        assert_series_equal(frame.select(r=dist.pmf(pl.col("x")))["r"], expected, check_exact=True)


def test_one_constant_beside_a_mismatched_column_still_raises() -> None:
    """One length-1 parameter does not make the call constant; the other is still aligned, and reported.

    Which layer reports the mismatch moves with the polars version, as in
    `broadcast_test.py::test_mismatched_lengths_raise`, so the message is matched loosely.
    """
    frame = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    mismatched = Binomial(n=pl.lit(5, dtype=_INT), p=pl.Series("p", [0.5, 0.5, 0.5])).pmf(pl.col("x"))

    with pytest.raises(pl.exceptions.PolarsError, match=r"incompatible lengths|non-equal length"):
        frame.select(mismatched)
