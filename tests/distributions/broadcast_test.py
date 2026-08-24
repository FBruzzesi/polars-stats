"""Row-alignment contract: a length-1 input expression broadcasts, it does not drop rows."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from polars_stats import Binomial, Normal
from polars_stats.distributions._base import ContinuousDistribution, DiscreteDistribution
from tests._polars_compat import (
    PARTITIONED_BROADCAST_AVAILABLE,
    arr_explode,
    assert_series_equal,
    linear_space,
)
from tests.property._specs import ALL_SPECS, SERIES_ROWS, ULP_ABS_TOL, ULP_REL_TOL

if TYPE_CHECKING:
    from collections.abc import Callable

    from polars_stats.distributions._base import _UnivariateDistribution
    from tests.property._specs import DistSpec

_ROWS = 4096
_GROUPS = 4
"""`_ROWS` is divisible by it, so no partition is empty."""

_VALUE_METHODS: tuple[tuple[str, str], ...] = (
    ("density", "x"),
    ("log_density", "x"),
    ("cdf", "x"),
    ("log_cdf", "x"),
    ("sf", "x"),
    ("log_sf", "x"),
    ("ppf", "q"),
    ("isf", "q"),
)
"""Every value-keyed method, with the column whose domain it accepts (`x` a support point, `q` a quantile)."""

_SAMPLERS: tuple[tuple[str, None], ...] = (("sample", None), ("samples", None))
"""The two sampler funnels. `None` marks "no value argument", which is also the bit-exact axis."""

_ALL_METHODS = _VALUE_METHODS + _SAMPLERS


def _frame(spec: DistSpec) -> pl.DataFrame:
    """A `_ROWS`-row frame: support points in the spec's evaluation window, quantiles, a partition key."""
    lo, hi = spec.eval_range(spec.example)
    return pl.DataFrame(
        {
            "x": linear_space(lo, hi, _ROWS),
            "q": linear_space(1e-3, 1.0 - 1e-3, _ROWS),
            "g": pl.int_range(0, _ROWS, eager=True) % _GROUPS,
        }
    )


_FRAMES = {spec.name: _frame(spec) for spec in ALL_SPECS}
"""One frame per spec, built once: no test mutates it."""

_needs_partitioned_broadcast = pytest.mark.skipif(
    not PARTITIONED_BROADCAST_AVAILABLE,
    reason="polars < 1.34 mishandles a length-1 input inside over / group_by().agg()",
)
"""Gate for the two partition suites below. The plain-`select` suites run on every supported version."""


def _call(spec: DistSpec, dist: _UnivariateDistribution, method: str, value: pl.Expr | None) -> pl.Expr:
    """One public method by name: a value-keyed call on `value`, or a seeded draw when `value` is `None`."""
    if value is None:
        return dist.sample(seed=0) if method == "sample" else dist.samples(size=3, seed=0)
    if method == "density":
        return spec.density(dist, value)
    if method == "log_density":
        if isinstance(dist, ContinuousDistribution):
            return dist.log_pdf(value)
        if isinstance(dist, DiscreteDistribution):
            return dist.log_pmf(value)
        msg = f"unsupported distribution family: {type(dist)}"  # pragma: no cover
        raise TypeError(msg)  # pragma: no cover
    call: Callable[[pl.Expr], pl.Expr] = getattr(dist, method)
    return call(value)


def _assert_same_rows(frame: pl.DataFrame, got: pl.Expr, want: pl.Expr, *, height: int, exact: bool = False) -> None:
    """`got` and `want` evaluate to the same `height` rows, element for element."""
    left = frame.select(got.alias("r"))["r"]
    right = frame.select(want.alias("r"))["r"]
    assert left.len() == height, f"expected {height} rows, got {left.len()}"
    assert_series_equal(left, right, check_exact=exact, rel_tol=ULP_REL_TOL, abs_tol=ULP_ABS_TOL)


def _assert_same_partitions(frame: pl.DataFrame, got: pl.Expr, want: pl.Expr, *, exact: bool = False) -> None:
    """`got` and `want` agree under both partition contexts, where `pl.len()` is the partition's length."""
    assert_series_equal(
        frame.with_columns(r=got.over("g"))["r"],
        frame.with_columns(r=want.over("g"))["r"],
        check_exact=exact,
        rel_tol=ULP_REL_TOL,
        abs_tol=ULP_ABS_TOL,
    )
    grouped = frame.group_by("g", maintain_order=True).agg(got=got, want=want)
    assert_series_equal(
        grouped["got"], grouped["want"], check_names=False, check_exact=exact, rel_tol=ULP_REL_TOL, abs_tol=ULP_ABS_TOL
    )


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.name)
@pytest.mark.parametrize(("method", "column"), _ALL_METHODS)
def test_length_one_parameters_broadcast(spec: DistSpec, method: str, column: str | None) -> None:
    """Length-1 parameters give the same rows as the same constants at full length, on every method."""
    frame = _FRAMES[spec.name]
    value = None if column is None else pl.col(column)
    broadcast, full_length = spec.make_literals(spec.example), spec.make_columns(spec.example)

    _assert_same_rows(
        frame,
        _call(spec, broadcast, method, value),
        _call(spec, full_length, method, value),
        height=_ROWS,
        exact=column is None,
    )


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.name)
@pytest.mark.parametrize(("method", "column"), _VALUE_METHODS)
def test_length_one_value_broadcasts(spec: DistSpec, method: str, column: str) -> None:
    """A length-1 *value* gives the same rows as the same value at full length, on every method."""
    frame = _FRAMES[spec.name]
    dist = spec.make_columns(spec.example)
    scalar = frame[column][0]

    _assert_same_rows(
        frame,
        _call(spec, dist, method, pl.lit(scalar)),
        _call(spec, dist, method, pl.repeat(scalar, n=pl.len())),
        height=_ROWS,
    )


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.name)
def test_sampler_draws_differ_per_row_under_literals(spec: DistSpec) -> None:
    """All-literal parameters still draw *per row*, and draw what the fast path draws.

    The load-bearing test in this file: a broadcast applied to the plugin's *result* rather than its *inputs*
    would return a constant column at the right height and pass everything else here.
    """
    frame = _FRAMES[spec.name]
    drawn = frame.select(r=spec.make_literals(spec.example).sample(seed=0))["r"]
    assert drawn.n_unique() > 1, "a broadcast result would make every row identical"
    _assert_same_rows(
        frame,
        spec.make_literals(spec.example).sample(seed=0),
        spec.make(spec.example).sample(seed=0),
        height=_ROWS,
        exact=True,
    )


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.name)
@pytest.mark.parametrize("method", ["sample", "samples"])
def test_sampler_reseeds_when_a_parameter_outruns_the_frame(spec: DistSpec, method: str) -> None:
    """A parameter longer than the frame sets the row count, and every row still gets its own seed.

    The row index is the *shorter* input here, and broadcasting it like a parameter would return one
    draw repeated at the right height, with no error.
    """
    call: Callable[[_UnivariateDistribution], pl.Expr] = (
        (lambda d: d.sample(seed=0)) if method == "sample" else (lambda d: d.samples(size=2, seed=0))
    )
    one_row = pl.DataFrame({"a": [0]})
    drawn = one_row.select(r=call(spec.make_series(spec.example)))["r"]

    assert drawn.len() == SERIES_ROWS
    # `samples` returns `Array`, whose `n_unique` the oldest supported polars does not implement.
    per_draw = arr_explode(drawn) if method == "samples" else drawn
    assert per_draw.n_unique() > 1, "a broadcast row index would seed every row identically"

    # Bit-equal to the frame-shaped spelling, so the index really is `0..n` and not merely non-constant.
    full_length = pl.DataFrame({"a": range(SERIES_ROWS)})
    expected = full_length.select(r=call(spec.make_columns(spec.example)))["r"]
    assert_series_equal(drawn, expected, check_exact=True)


@pytest.mark.parametrize("reducer", ["first", "max"])
def test_reduced_value_expressions_broadcast(reducer: str) -> None:
    """A reduced value is length 1 too, so the alignment is by length, not by expression kind.

    `expr.meta.is_literal()` catches `pl.lit` and misses both of these. One distribution suffices: the
    spelling is what is under test.
    """
    frame = _FRAMES["normal"]
    dist = Normal(mu=pl.repeat(1.5, n=pl.len()), sigma=pl.repeat(2.0, n=pl.len()))
    scalar = getattr(frame["x"], reducer)()

    _assert_same_rows(
        frame,
        dist.cdf(getattr(pl.col("x"), reducer)()),
        dist.cdf(pl.repeat(scalar, n=pl.len())),
        height=_ROWS,
    )


@pytest.mark.parametrize("moment", ["mean", "variance", "std", "median", "entropy"])
def test_length_one_parameter_beside_a_column_moment(moment: str) -> None:
    """A length-1 parameter beside a full-length one gives a full-length moment.

    With *every* parameter length 1 the moment is legitimately a scalar column instead, which is `pl.lit`'s
    own semantics and is pinned by `tests/property/moment_test.py`.
    """
    mixed = Normal(mu=pl.lit(1.5), sigma=pl.repeat(2.0, n=pl.len()))
    full_length = Normal(mu=pl.repeat(1.5, n=pl.len()), sigma=pl.repeat(2.0, n=pl.len()))

    _assert_same_rows(_FRAMES["normal"], getattr(mixed, moment)(), getattr(full_length, moment)(), height=_ROWS)


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.name)
@pytest.mark.parametrize(("method", "column"), _ALL_METHODS)
def test_zero_row_frame_stays_empty(spec: DistSpec, method: str, column: str | None) -> None:
    """A length-1 input over a 0-row frame yields 0 rows, not 1.

    Every method here keeps one full-length input (the value column, or the sampler's row index), so this is
    not the all-constant case, which is one row by design.
    """
    empty = _FRAMES[spec.name].head(0)
    value = None if column is None else pl.col(column)

    assert empty.select(_call(spec, spec.make_literals(spec.example), method, value).alias("r")).height == 0


def test_mismatched_lengths_raise() -> None:
    """Lengths that are neither equal nor 1 have no defined broadcast, so they raise.

    *Which layer* rejects the shape is a polars scheduling detail that moves with the version and the engine:
    `align_inputs` names both lengths, while polars' own zip node reports `non-equal length inputs` when it
    gets there first. Both are correct rejections, so the message is matched loosely and the raise itself,
    which is the contract, is asserted strictly. `engine=` is deliberately not passed: `"in-memory"` is not a
    valid engine name on the oldest supported polars.
    """
    frame = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "p": [0.5] * 4})
    mismatched = Binomial(n=pl.Series("n", [1, 2, 3]), p=pl.col("p")).pmf(pl.col("x"))
    message = r"incompatible lengths|non-equal length"

    with pytest.raises(pl.exceptions.PolarsError, match=message):
        frame.select(mismatched)

    with pytest.raises(pl.exceptions.PolarsError, match=message):
        frame.lazy().select(mismatched).collect()


@_needs_partitioned_broadcast
@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.name)
def test_partition_contexts_broadcast_length_one_parameters(spec: DistSpec) -> None:
    """Under `over` and `group_by().agg()`, length-1 parameters broadcast to the *partition* length."""
    x = pl.col("x")
    _assert_same_partitions(
        _FRAMES[spec.name],
        spec.density(spec.make_literals(spec.example), x),
        spec.density(spec.make_columns(spec.example), x),
    )


@_needs_partitioned_broadcast
@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.name)
def test_partition_contexts_broadcast_a_partition_local_value(spec: DistSpec) -> None:
    """A value reduced *within* the partition is the case a fix written against whole-frame length survives."""
    dist = spec.make_columns(spec.example)
    reduced = pl.col("x").max()
    _assert_same_partitions(
        _FRAMES[spec.name], spec.density(dist, reduced), spec.density(dist, pl.repeat(reduced, n=pl.len()))
    )


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.name)
def test_partition_contexts_keep_sampler_bit_equality(spec: DistSpec) -> None:
    """The sampler funnel keeps its per-row seed under partitioning, where the row index is partition-local."""
    _assert_same_partitions(
        _FRAMES[spec.name],
        spec.make_literals(spec.example).sample(seed=0),
        spec.make_columns(spec.example).sample(seed=0),
        exact=True,
    )


def test_multi_chunk_frame_broadcasts() -> None:
    """A broadcast input is single-chunk; the column it is zipped against need not be.

    `try_*_elementwise` calls `align_chunks_*` itself and the `*_param_rows` iterators walk chunks in order,
    so length alignment is all the funnels owe. Nothing else pinned that.
    """
    parts = [pl.DataFrame({"x": [float(i) for i in range(k, k + 8)]}) for k in (0, 8, 16)]
    chunked = pl.concat(parts, rechunk=False)
    assert chunked.n_chunks() > 1

    broadcast = Binomial(n=pl.lit(20, dtype=pl.Int64), p=pl.lit(0.3)).pmf("x")
    full_length = Binomial(n=pl.repeat(20, n=pl.len(), dtype=pl.Int64), p=pl.repeat(0.3, n=pl.len())).pmf("x")
    _assert_same_rows(chunked, broadcast, full_length, height=chunked.height, exact=True)
    assert_series_equal(chunked.select(r=broadcast)["r"], chunked.rechunk().select(r=broadcast)["r"], check_exact=True)


@pytest.mark.parametrize(
    ("n", "message"),
    [
        (pl.lit(5.0), "n must be an integer column"),
        (pl.lit(-5, dtype=pl.Int64), "n must be a non-negative integer"),
    ],
)
def test_broadcast_n_still_rejects_a_bad_dtype(n: pl.Expr, message: str) -> None:
    """Alignment runs *before* the cast, so `coerce_n` judges the expanded column and still rejects it.

    `n` is the one parameter whose cast can reject, so it is the one place the order is observable.
    """
    frame = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "p": [0.5] * 4})

    with pytest.raises(pl.exceptions.ComputeError, match=message):
        frame.select(r=Binomial(n=n, p="p").pmf("x"))


def test_broadcast_n_propagates_nulls_and_widens() -> None:
    """A null length-1 `n` nulls every row, and a `UInt64` one passes the widening cast unchanged."""
    frame = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "p": [0.5] * 4})

    nulls = frame.select(r=Binomial(n=pl.lit(None, dtype=pl.Int64), p="p").pmf("x"))["r"]
    assert nulls.len() == frame.height
    assert nulls.null_count() == frame.height

    _assert_same_rows(
        frame,
        Binomial(n=pl.lit(5, dtype=pl.UInt64), p="p").pmf("x"),
        Binomial(n=pl.repeat(5, n=pl.len(), dtype=pl.UInt64), p="p").pmf("x"),
        height=frame.height,
        exact=True,
    )
