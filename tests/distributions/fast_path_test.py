"""The constant-parameter fast paths, and that they agree with the general per-row path.

All-scalar parameters route every value-keyed method through the ``<name>_<method>_scalar`` plugin and
every moment through a length-1 validator call, each validating once instead of per row.
`tests/property/value_keyed_test.py` and `moment_test.py` pin bit-equality against the per-row path for
*valid* parameters; this module pins what those cannot reach:

* both paths refuse the same parameterisations, and accept the same degenerate ones;
* constants validate up front and columns value by value, so an invalid constant raises on a zero-row
  frame while an invalid empty column returns empty;
* one constant beside a column still aligns, and a mismatched column is still reported;
* a null row, in any input and in any chunk, nulls that row and leaves every other row bit for bit as
  the null-free call answers it.

A Python `None` parameter and a negative scalar `n` are rejected at construction, so neither has a
case here. The invalid parameterisations come from `tests/_registry.invalid_cases`, the same list
`validation_test.py` sweeps.
"""

from __future__ import annotations

from math import inf, nan
from typing import TYPE_CHECKING, Literal

import polars as pl
import pytest

from polars_stats import Binomial, DiscreteUniform, Normal
from tests._polars_compat import ENGINE_SELECTABLE, assert_series_equal
from tests._registry import (
    ALL_SPECS,
    CONTRACT_FRAME,
    DEGENERATE,
    DRIVER_REGIMES,
    SPECS_BY_NAME,
    all_value_keyed,
    constant_column,
    density,
    invalid_cases,
    point_column,
)

if TYPE_CHECKING:
    from polars_stats.distributions._base import _UnivariateDistribution
    from tests._registry import DistSpec, InvalidCase, Method

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
    """A null length-1 parameter nulls every row of every value-keyed method, as a null parameter column does."""
    frame = pl.DataFrame({"x": [0.5, 1.0, 2.0], "q": [0.1, 0.5, 0.9]})
    dist = _null_constant(spec)

    for method in all_value_keyed(dist):
        result = frame.select(r=getattr(dist, method)(pl.col(point_column(method))))["r"]

        assert result.len() == frame.height, method
        assert result.null_count() == frame.height, method


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


_POINTS = pl.DataFrame(
    {
        # Out of `[0, 1]`, so the inverses answer null, at rows 0, 4, 8 and 9.
        "q": [1.5, 0.1, nan, 0.5, -0.5, 0.9, 0.0, 1.0, inf, 2.0],
        "x": [0.5, nan, -1.0, 1.5, 3.0, inf, -inf, 0.0, 7.0, 2.0],
    }
)

_HALF = _POINTS.height // 2

_LAYOUTS: dict[str, tuple[slice | Literal["point", "param", "all"], ...]] = {
    # Null rows before, between and after the data: in the point only, in one parameter only, in all.
    "nulls-around": ("point", slice(None, _HALF), "param", slice(_HALF, None), "all"),
    # Rows 1 to 3 answer for every method, so the validity bitmap first appears in a later chunk: at
    # `q = 1.5` for the inverses, at the null parameter for the rest.
    "null-free-first": (slice(1, 4), slice(None), "param"),
}
"""Each layout's pieces in order: a slice of the null-free frame, or a null row named by what it nulls."""

_FILLER = slice(3, 4)
"""The null-free row a null row copies its other columns from: every method answers there, so the row is null
only because of what it nulls."""


def _null_free_frame() -> pl.DataFrame:
    """`CONTRACT_FRAME`'s parameters cycled beside `_POINTS`, each column split into two chunks at its own row.

    The split row cycles with the column index, so the columns of each call disagree on at least one
    boundary, and a loop that zips the inputs' chunks without aligning them first drops or misreads rows.
    """
    rows = _POINTS.height
    params = CONTRACT_FRAME.drop(_POINTS.columns)[[i % CONTRACT_FRAME.height for i in range(rows)]]
    columns = [*params.get_columns(), *_POINTS.get_columns()]
    return pl.DataFrame(
        [pl.concat([c[: 1 + i % (rows - 1)], c[1 + i % (rows - 1) :]], rechunk=False) for i, c in enumerate(columns)]
    )


_VALUE_KEYED_CASES = [
    pytest.param(spec, method, id=f"{spec.name}.{method}")
    for spec in ALL_SPECS
    for method in all_value_keyed(spec.build("scalar"))
]


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
@pytest.mark.parametrize("aligned", [True, False], ids=["aligned-chunks", "staggered-chunks"])
@pytest.mark.parametrize("layout", _LAYOUTS)
@pytest.mark.parametrize(("spec", "method"), _VALUE_KEYED_CASES)
def test_null_rows_leave_other_rows_unchanged(
    spec: DistSpec, method: Method, layout: str, engine: Literal["in-memory", "streaming"], *, aligned: bool
) -> None:
    """Each `_LAYOUTS` row is null in a null-row piece and bit for bit the null-free answer elsewhere.

    Aligned chunks share their boundaries across inputs; staggered chunks are what the drivers must align
    before zipping. `pl.len()` beside the call keeps the in-memory engine from splitting the frame into
    one call per chunk, so a single call reads every chunk in order.
    """
    frame = _null_free_frame()
    point = point_column(method)
    expr = getattr(spec.build("name"), method)(pl.col(point))
    nulled_by = {"point": {point}, "param": {spec.parameters[-1].column}, "all": set(frame.columns)}

    def null_row(nulled: set[str]) -> pl.DataFrame:
        return pl.DataFrame(
            [pl.Series(n, [None] if n in nulled else frame[n][_FILLER], d) for n, d in frame.schema.items()]
        )

    def rows(part: slice) -> pl.DataFrame:
        return frame[part].rechunk() if aligned else frame[part]

    pieces = _LAYOUTS[layout]
    combined = pl.concat([rows(p) if isinstance(p, slice) else null_row(nulled_by[p]) for p in pieces], rechunk=False)
    boundaries = {tuple(map(len, combined[c].get_chunks())) for c in [point, *(p.column for p in spec.parameters)]}
    assert combined[point].n_chunks() >= len(pieces)  # guard against a silent rechunk
    assert (len(boundaries) == 1) is aligned  # staggered inputs disagree on at least one boundary
    query = combined.lazy().select(expr, pl.len())
    with_nulls = (query.collect(engine=engine) if ENGINE_SELECTABLE else query.collect()).to_series()

    null_free = frame.select(expr).to_series()
    assert null_free[1:4].null_count() == 0  # the "null-free-first" premise
    gap = pl.Series(null_free.name, [None], null_free.dtype)
    expected = pl.concat([null_free[p] if isinstance(p, slice) else gap for p in pieces], rechunk=False)
    assert_series_equal(with_nulls, expected, check_exact=True)


_OFF_GRID = 3
"""Rows in the null-free first chunk, so every later chunk's validity lands mid-byte in the output."""

_SCATTERED: tuple[tuple[Literal["point", "param"], tuple[int, ...]], ...] = (("point", (2, 5)), ("param", (5, 7)))
"""After the first chunk, one chunk per input with that input null at these rows and present elsewhere."""


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
@pytest.mark.parametrize(("spec", "method"), _VALUE_KEYED_CASES)
def test_scattered_null_inputs_in_later_chunks_leave_other_rows_unchanged(
    spec: DistSpec, method: Method, engine: Literal["in-memory", "streaming"]
) -> None:
    """Chunks mixing null and present rows of one input, after a null-free chunk, null only those rows.

    Each later chunk is a slice one row in, so its validity bitmap starts mid-byte, and only one input
    carries it, so the driver's combined validity is that bitmap with its offset rather than a fresh one.
    """
    frame = _null_free_frame().rechunk()
    point = point_column(method)
    column = {"point": point, "param": spec.parameters[-1].column}
    row = pl.int_range(pl.len())

    def nulled(input_: str, rows: tuple[int, ...]) -> pl.DataFrame:
        c = column[input_]
        return frame.with_columns(pl.when(row.is_in(rows)).then(None).otherwise(pl.col(c)).alias(c))[1:]

    pieces = [frame[:_OFF_GRID], *(nulled(input_, rows) for input_, rows in _SCATTERED)]
    combined = pl.concat(pieces, rechunk=False)
    assert combined[point].n_chunks() == len(pieces)  # guard against a silent rechunk
    expr = getattr(spec.build("name"), method)(pl.col(point))
    query = combined.lazy().select(expr, pl.len())
    with_nulls = (query.collect(engine=engine) if ENGINE_SELECTABLE else query.collect()).to_series()

    null_free = frame.select(expr)
    name = null_free.columns[0]
    masked = (
        null_free.select(pl.when(row.is_in(rows)).then(None).otherwise(name).alias(name))[1:] for _, rows in _SCATTERED
    )
    expected = pl.concat([null_free[:_OFF_GRID], *masked], rechunk=False).to_series()
    assert_series_equal(with_nulls, expected, check_exact=True)
