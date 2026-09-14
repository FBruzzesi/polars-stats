"""The constant-parameter fast paths, and that they agree with the general per-row path.

All-scalar parameters route every value-keyed method through the ``<name>_<method>_scalar`` plugin and
every moment through a length-1 validator call, each validating once instead of per row.
`tests/property/value_keyed_test.py` and `moment_test.py` pin bit-equality against the per-row path for
*valid* parameters; this module pins what those cannot reach:

* both paths refuse the same parameterisations, and accept the same degenerate ones;
* constants validate up front and columns value by value, so an invalid constant raises on a zero-row
  frame while an invalid empty column returns empty;
* one constant beside a column still aligns, and a mismatched column is still reported.

A Python `None` parameter and a negative scalar `n` are rejected at construction, so neither has a
case here. The invalid parameterisations come from `tests/_registry.invalid_cases`, the same list
`validation_test.py` sweeps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from polars_stats import Binomial, DiscreteUniform, Normal
from tests._polars_compat import assert_series_equal
from tests._registry import (
    ALL_SPECS,
    DEGENERATE,
    DRIVER_REGIMES,
    SPECS_BY_NAME,
    constant_column,
    density,
    invalid_cases,
)

if TYPE_CHECKING:
    from polars_stats.distributions._base import _UnivariateDistribution
    from tests._registry import DistSpec, InvalidCase

INVALID = [pytest.param(spec, case, id=f"{spec.name} {case.label}") for spec, case in invalid_cases(ALL_SPECS)]
"""Every parameterisation both paths must refuse."""


@pytest.mark.parametrize(("spec", "case"), INVALID)
def test_the_value_keyed_paths_agree_on_validation(spec: DistSpec, case: InvalidCase) -> None:
    """The fast path and the per-row path agree on what is a valid parameterisation, and say why."""
    frame = pl.DataFrame({"x": [0.5, 1.0, 2.0]})

    for regime in DRIVER_REGIMES:
        with pytest.raises(pl.exceptions.ComputeError, match=case.match):
            frame.select(r=density(spec.build(regime, case.params), pl.col("x")))


@pytest.mark.parametrize(("spec", "case"), INVALID)
def test_the_moment_paths_agree_on_validation(spec: DistSpec, case: InvalidCase) -> None:
    """`variance` is the representative moment: it routes through the validator for every distribution."""
    frame = pl.DataFrame({"_": range(4)})

    for regime in DRIVER_REGIMES:
        with pytest.raises(pl.exceptions.ComputeError, match=case.match):
            frame.select(r=spec.build(regime, case.params).variance())


_DEGENERATE_CASES = [
    pytest.param(SPECS_BY_NAME[name], params, id=label) for label, (name, params) in DEGENERATE.items()
]


@pytest.mark.parametrize(("spec", "params"), _DEGENERATE_CASES)
def test_degenerate_valid_params_agree_across_paths(spec: DistSpec, params: tuple[float, ...]) -> None:
    """A mass-collapsing endpoint is a legitimate parameterisation; neither path may refuse it."""
    frame = pl.DataFrame({"x": [0.5, 1.0, 2.0]})
    fast = frame.select(r=density(spec.build("scalar", params), pl.col("x")))["r"]
    per_row = frame.select(r=density(spec.build("column", params), pl.col("x")))["r"]
    assert_series_equal(fast, per_row, check_exact=True)


_INT = pl.Int64()


def test_constant_parameters_validate_on_empty_input(spec: DistSpec) -> None:
    """An invalid constant raises with no rows to observe it; an invalid empty column returns empty.

    One case per distribution, taken from the head of its `invalid` table, which is one per Rust
    driver by construction.
    """
    empty = pl.DataFrame({"x": []}, schema={"x": pl.Float64})
    case = spec.invalid[0]

    for regime in ("scalar", "literal"):
        with pytest.raises(pl.exceptions.ComputeError, match=case.match):
            empty.select(r=density(spec.build(regime, case.params), pl.col("x")))

    assert empty.select(r=density(spec.build("column", case.params), pl.col("x"))).height == 0


def test_a_constant_the_cast_rejects_also_raises_on_an_empty_frame() -> None:
    """`n = -5` is refused by the strict `Int64 -> UInt64` cast, not a domain check, and still before any row.

    A negative Python-scalar `n` has no spelling here: `coerce_n` rejects it at construction.
    """
    empty = pl.DataFrame({"x": []}, schema={"x": pl.Float64})

    with pytest.raises(pl.exceptions.PolarsError, match="n must be a non-negative integer"):
        empty.select(r=Binomial(n=pl.lit(-5, dtype=_INT), p=pl.lit(0.5)).pmf(pl.col("x")))

    assert empty.select(r=Binomial(n=constant_column(-5, _INT), p=constant_column(0.5)).pmf(pl.col("x"))).height == 0


def _null_constant(spec: DistSpec) -> _UnivariateDistribution:
    """The distribution with its *first* parameter a null length-1 literal and the rest valid."""
    first = spec.parameters[0]
    kwargs: dict[str, object] = {first.name: pl.lit(None, dtype=first.dtype)}
    kwargs.update({p.name: pl.lit(p.cast(v), dtype=p.dtype) for p, v in spec.pairs()[1:]})
    return spec.cls(**kwargs)


def test_a_null_constant_parameter_nulls_every_row(spec: DistSpec) -> None:
    """A null length-1 parameter nulls every row, as a null parameter column does."""
    frame = pl.DataFrame({"x": [0.5, 1.0, 2.0]})

    result = frame.select(r=density(_null_constant(spec), pl.col("x")))["r"]

    assert result.len() == frame.height
    assert result.null_count() == frame.height


def test_a_null_constant_parameter_still_gates_the_value_dtype(spec: DistSpec) -> None:
    """A null length-1 parameter does not skip the value column's dtype gate: a `String` column is still refused."""
    strings = pl.DataFrame({"x": ["a", "b"]})

    with pytest.raises(pl.exceptions.ComputeError, match="must be a numeric column"):
        strings.select(r=density(_null_constant(spec), pl.col("x")))


@pytest.mark.parametrize("dtype", [pl.Int64, pl.UInt64, pl.Int32])
def test_discreteuniform_integer_points_agree_across_bound_spellings(dtype: pl.DataType) -> None:
    """Integer points stay exact under every bound spelling; `UInt64` is the one dtype with its own arm."""
    frame = pl.DataFrame({"x": [0, 1, 3, 6, 7]}, schema={"x": dtype})
    expected = frame.select(r=DiscreteUniform(1, 6).pmf(pl.col("x")))["r"]

    literal = DiscreteUniform(pl.lit(1, dtype=_INT), pl.lit(6, dtype=_INT))
    column = DiscreteUniform(constant_column(1, _INT), constant_column(6, _INT))
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


@pytest.mark.parametrize(
    ("scalar", "per_row"),
    [(Binomial(5, 1.5), Binomial(constant_column(5, pl.Int64()), constant_column(1.5)))],
    ids=["p=1.5"],
)
def test_binomial_entropy_scalar_and_column_paths_agree_on_validation(scalar: Binomial, per_row: Binomial) -> None:
    """`Binomial.entropy` is its own plugin (the Rust support sum), so its validation is pinned on its own."""
    frame = pl.DataFrame({"_": range(4)})
    for dist in (scalar, per_row):
        with pytest.raises(pl.exceptions.ComputeError, match="p must be in"):
            frame.select(r=dist.entropy())


def test_moment_fast_path_on_empty_frame_is_a_scalar() -> None:
    """On a zero-row frame the scalar moment is still one row, and still validates.

    Both follow from the scalar path being built from length-1 literals, exactly as
    `df.head(0).select(pl.lit(1.0))` is one row. The per-row path's column pass sees no value on an
    empty frame, so it returns empty without raising.
    """
    empty = pl.DataFrame({"_": []}, schema={"_": pl.Int64})

    valid_fast = empty.select(r=Normal(0.0, 2.0).variance())
    valid_slow = empty.select(r=Normal(constant_column(0.0), constant_column(2.0)).variance())
    assert valid_fast.height == 1
    assert valid_fast["r"].to_list() == [4.0]
    assert valid_slow.height == 0

    with pytest.raises(pl.exceptions.ComputeError):
        empty.select(r=Normal(0.0, -1.0).variance())
    assert empty.select(r=Normal(constant_column(0.0), constant_column(-1.0)).variance()).height == 0


@pytest.mark.parametrize(("spec", "params"), _DEGENERATE_CASES)
def test_degenerate_valid_params_entropy_is_zero(spec: DistSpec, params: tuple[float, ...]) -> None:
    """All the mass on one point means no uncertainty, so every one of these is exactly `0`.

    Not a per-row expectation: the value is `0` for every case here by definition, and a table column
    that never varies is a constant wearing a row's clothes. `Binomial` is the one to watch, since
    its entropy is a Rust support sum rather than a closed form.
    """
    frame = pl.DataFrame({"_": range(4)})
    scalar, per_row = spec.build("scalar", params), spec.build("column", params)

    # On its own the fast path is a scalar column: one row, whatever the frame height.
    assert frame.select(r=scalar.entropy())["r"].to_list() == [0.0]
    both = frame.select(fast=scalar.entropy(), slow=per_row.entropy())
    assert both["slow"].to_list() == [0.0] * frame.height
    assert_series_equal(both["fast"], both["slow"], check_names=False, check_exact=True)
