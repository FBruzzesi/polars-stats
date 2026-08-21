from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from polars_stats import Binomial
from tests._polars_compat import assert_series_equal

if TYPE_CHECKING:
    from collections.abc import Callable

# Every public method must report an invalid parameter as a ComputeError, not silently null the row.
# All deterministic methods build the statrs distribution in their plugin; the sampler validates in
# its own plugin.
_METHODS: dict[str, Callable[[Binomial], pl.Expr]] = {
    "pmf": lambda b: b.pmf(pl.col("x")),
    "log_pmf": lambda b: b.log_pmf(pl.col("x")),
    "cdf": lambda b: b.cdf(pl.col("x")),
    "log_cdf": lambda b: b.log_cdf(pl.col("x")),
    "sf": lambda b: b.sf(pl.col("x")),
    "log_sf": lambda b: b.log_sf(pl.col("x")),
    "ppf": lambda b: b.ppf(pl.col("q")),
    "isf": lambda b: b.isf(pl.col("q")),
    "mean": lambda b: b.mean(),
    "variance": lambda b: b.variance(),
    "std": lambda b: b.std(),
    "median": lambda b: b.median(),
    "entropy": lambda b: b.entropy(),
    "sample": lambda b: b.sample(seed=0),
    "samples": lambda b: b.samples(size=4, seed=0),
}


@pytest.mark.parametrize("expr_fn", _METHODS.values(), ids=list(_METHODS))
def test_method_raises_on_out_of_range_p_column(expr_fn: Callable[[Binomial], pl.Expr]) -> None:
    df = pl.DataFrame({"n": [10, 10], "p": [0.5, 1.5], "x": [0, 0], "q": [0.5, 0.5]})  # row 1: p out of [0, 1]
    b = Binomial(pl.col("n"), pl.col("p"))
    with pytest.raises(pl.exceptions.ComputeError, match="p must be in"):
        df.select(r=expr_fn(b))


@pytest.mark.parametrize("expr_fn", _METHODS.values(), ids=list(_METHODS))
def test_method_raises_on_out_of_range_p_scalar(expr_fn: Callable[[Binomial], pl.Expr]) -> None:
    df = pl.DataFrame({"x": [0], "q": [0.5]})
    b = Binomial(10, 1.5)
    with pytest.raises(pl.exceptions.ComputeError, match="p must be in"):
        df.select(r=expr_fn(b))


@pytest.mark.parametrize("expr_fn", _METHODS.values(), ids=list(_METHODS))
def test_method_raises_on_negative_n_column(expr_fn: Callable[[Binomial], pl.Expr]) -> None:
    df = pl.DataFrame({"n": [10, -1], "p": [0.5, 0.5], "x": [0, 0], "q": [0.5, 0.5]})  # row 1: n < 0
    b = Binomial(pl.col("n"), pl.col("p"))
    with pytest.raises(pl.exceptions.ComputeError, match="n must be a non-negative integer"):
        df.select(r=expr_fn(b))


# A `Float64` `n` used to truncate inside the plugins while the Python closed-form moments kept the
# fraction, so `n = 2.7` gave `pmf` for `n = 2` beside `mean` for `n = 2.7`, raising nothing.
_NOT_INTEGER = "n must be an integer column"
_NEGATIVE = "n must be a non-negative integer"

# `2.0` is refused like `2.7`, because the check reads the dtype and not the value. A value-level rule
# would refuse the same input the scalar path already refuses on type, but for a different reason.
_FLOAT_NS = [2.7, 2.0]

_INT_DTYPES: list[type[pl.DataType]] = [
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
]
_SIGNED_DTYPES = [dtype for dtype in _INT_DTYPES if dtype.is_signed_integer()]


def _dtype_ids(dtypes: list[type[pl.DataType]]) -> list[str]:
    return [str(dtype()) for dtype in dtypes]


def _float_n_frame(n: float) -> pl.DataFrame:
    return pl.DataFrame(
        {"n": [10.0, n], "p": [0.5, 0.5], "x": [0, 2], "q": [0.5, 0.5]},
        schema={"n": pl.Float64, "p": pl.Float64, "x": pl.Int64, "q": pl.Float64},
    )


@pytest.mark.parametrize("n", _FLOAT_NS)
@pytest.mark.parametrize("expr_fn", _METHODS.values(), ids=list(_METHODS))
def test_method_raises_on_float_n_column(expr_fn: Callable[[Binomial], pl.Expr], n: float) -> None:
    b = Binomial(pl.col("n"), pl.col("p"))
    with pytest.raises(pl.exceptions.ComputeError, match=_NOT_INTEGER):
        _float_n_frame(n).select(r=expr_fn(b))


@pytest.mark.parametrize("n", _FLOAT_NS)
def test_float_n_cannot_split_a_single_query(n: float) -> None:
    # One `select`: a fix that made only `pmf` raise would leave `mean` / `variance` evaluating a
    # different distribution from the same parameters.
    df = pl.DataFrame({"n": [n], "p": [0.5]})
    b = Binomial(n=pl.col("n"), p=pl.col("p"))
    with pytest.raises(pl.exceptions.ComputeError, match=_NOT_INTEGER):
        df.select(pmf=b.pmf(pl.lit(2.0)), mean=b.mean(), var=b.variance())


@pytest.mark.parametrize("n", _FLOAT_NS)
def test_float_n_scalar_raises(n: float) -> None:
    # Both paths refuse a float; the wider type matrix is in `construct_test.py`.
    with pytest.raises(TypeError, match="n should be an int or IntoExprColumn"):
        Binomial(n=n, p=0.5)  # type: ignore[arg-type]


@pytest.mark.parametrize("expr_fn", _METHODS.values(), ids=list(_METHODS))
def test_explicit_cast_is_the_supported_route(expr_fn: Callable[[Binomial], pl.Expr]) -> None:
    # The error message tells the caller to write this, so it has to hold, draws included.
    data = {"p": [0.5, 0.25], "x": [0, 2], "q": [0.5, 0.5]}
    native = pl.DataFrame({"n": [10, 3], **data}, schema_overrides={"n": pl.Int64})
    floats = pl.DataFrame({"n": [10.0, 3.0], **data}, schema_overrides={"n": pl.Float64})

    assert_series_equal(
        floats.select(r=expr_fn(Binomial(pl.col("n").cast(pl.Int64), pl.col("p"))))["r"],
        native.select(r=expr_fn(Binomial(pl.col("n"), pl.col("p"))))["r"],
        check_dtypes=True,
    )


def test_the_whole_u64_range_of_n_is_accepted() -> None:
    # `n` reaches `statrs` as the `u64` it is there, so a count above `i64::MAX` is a count like any
    # other; the `Int64` funnel mapped the upper half of the range to `null`. `mean` and `sample`
    # only: `entropy` would sum the whole `{0, ..., n}` support.
    trials = [2**63, 2**64 - 1]
    df = pl.DataFrame({"n": trials, "p": [0.5, 0.5]}, schema={"n": pl.UInt64, "p": pl.Float64})
    b = Binomial(n=pl.col("n"), p=pl.col("p"))

    got = df.select(mean=b.mean(), draw=b.sample(seed=0))
    assert got["mean"].to_list() == pytest.approx([n * 0.5 for n in trials])
    assert got["draw"].dtype == pl.UInt64
    assert all(0 <= drawn <= n for drawn, n in zip(got["draw"].to_list(), trials, strict=True))


@pytest.mark.parametrize("dtype", _SIGNED_DTYPES, ids=_dtype_ids(_SIGNED_DTYPES))
def test_negative_n_raises_at_every_signed_width(dtype: type[pl.DataType]) -> None:
    # A negative has no `u64` to widen to, so the strict widening is the sign check. Polars' overflow
    # message carries the offending value, so the error still names it.
    df = pl.DataFrame({"n": [5, -1], "p": [0.5, 0.5]}, schema={"n": dtype, "p": pl.Float64})
    with pytest.raises(pl.exceptions.ComputeError, match=_NEGATIVE) as raised:
        df.select(r=Binomial(pl.col("n"), pl.col("p")).mean())
    assert "-1" in str(raised.value)


def test_negative_n_is_judged_for_the_column_not_the_row() -> None:
    # `n` is checked once for the column where `p` is checked per row, so a null `p` no longer hides a
    # negative count. A null `n` is still a null row.
    negative = pl.DataFrame({"n": [-1], "p": [None]}, schema={"n": pl.Int64, "p": pl.Float64})
    with pytest.raises(pl.exceptions.ComputeError, match=_NEGATIVE):
        negative.select(r=Binomial(pl.col("n"), pl.col("p")).mean())

    null_n = pl.DataFrame({"n": [None], "p": [0.5]}, schema={"n": pl.Int64, "p": pl.Float64})
    assert null_n.select(r=Binomial(pl.col("n"), pl.col("p")).mean())["r"].to_list() == [None]


@pytest.mark.parametrize("dtype", _INT_DTYPES, ids=_dtype_ids(_INT_DTYPES))
def test_every_integer_width_is_accepted(dtype: type[pl.DataType]) -> None:
    # Any integer width widens to `UInt64`, so the rule is integrality and not `Int64` specifically.
    df = pl.DataFrame({"n": [10], "p": [0.5], "x": [1.0]}, schema={"n": dtype, "p": pl.Float64, "x": pl.Float64})
    b = Binomial(pl.col("n"), pl.col("p"))
    got = df.select(pmf=b.pmf(pl.col("x")), mean=b.mean(), var=b.variance())
    assert got.row(0) == pytest.approx((10 * 0.5**10, 5.0, 2.5))


def test_null_dtype_n_column_propagates() -> None:
    # A `Null`-dtype column has no values the integer rule could protect, so it stays
    # null-in-null-out like every other parameter instead of hitting the dtype gate.
    df = pl.DataFrame({"n": [None, None], "p": [0.5, 0.5]})  # n dtype: Null
    b = Binomial(pl.col("n"), pl.col("p"))
    # `pmf(1.0)`, not `pmf(pl.lit(1.0))`: a Python scalar is row-aligned by `_coerce`'s
    # `pl.repeat`, where a caller-built length-1 literal expr is passed to the plugin as-is and
    # `try_ternary_elementwise` zips it to length 1 (the truncation `_coerce` documents). The
    # in-memory engine exposes that; the streaming engine broadcasts upstream and hides it.
    got = df.select(mean=b.mean(), pmf=b.pmf(1.0), draw=b.sample(seed=0))
    assert got["mean"].to_list() == [None, None]
    assert got["pmf"].to_list() == [None, None]
    assert got["draw"].to_list() == [None, None]


def test_all_null_float_n_column_still_raises_on_dtype() -> None:
    # The dtype rule is column-level: a float `n` column raises even when every value in it is null.
    df = pl.DataFrame({"n": [None, None], "p": [0.5, 0.5]}, schema={"n": pl.Float64, "p": pl.Float64})
    with pytest.raises(pl.exceptions.ComputeError, match=_NOT_INTEGER):
        df.select(r=Binomial(pl.col("n"), pl.col("p")).mean())


def test_entropy_raises_at_the_u64_maximum() -> None:
    # statrs evaluates entropy by iterating `(0..n + 1)`, which wraps to an empty range at
    # `u64::MAX` in release and returned a confident 0.0 (true value ~22.9 nats). The one value
    # `test_the_whole_u64_range_of_n_is_accepted` cannot extend to entropy, refused loudly.
    df = pl.DataFrame({"n": [2**64 - 1], "p": [0.5]}, schema={"n": pl.UInt64, "p": pl.Float64})
    with pytest.raises(pl.exceptions.ComputeError, match="overflows the entropy support sum"):
        df.select(r=Binomial(pl.col("n"), pl.col("p")).entropy())
