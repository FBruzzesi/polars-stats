from __future__ import annotations

import polars as pl
import pytest

from polars_stats import Binomial


@pytest.mark.parametrize("bad_n", [None, True, False, 1.5, 2.0, [5], (5,), {"n": 5}])
def test_construct_invalid_n_type_raises(bad_n: object) -> None:
    with pytest.raises(TypeError, match="n should be an int or IntoExprColumn"):
        Binomial(n=bad_n, p=0.5)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_p", [None, True, False, 1, 0, [0.5], (0.5,), {"p": 0.5}])
def test_construct_invalid_p_type_raises(bad_p: object) -> None:
    with pytest.raises(TypeError, match="p should be a float or IntoExprColumn"):
        Binomial(n=5, p=bad_p)  # type: ignore[arg-type]


@pytest.mark.parametrize("n", [0, 1, 10, 2**63 - 1, "n", pl.col("n"), pl.Series("n", [5])])
def test_construct_accepts_valid_n_types(n: object) -> None:
    # int and IntoExprColumn are accepted; only `n`'s own bounds are checked here, not `p`'s range.
    Binomial(n=n, p=0.5)  # type: ignore[arg-type]


# `n` is the one parameter whose *value* is judged at construction: a Python `int` is expanded to a
# `UInt64` column and rides the fast paths as a kwarg, so neither a negative nor a count past the
# kwargs wire's `i64` can be reported from the plugin the way an out-of-range `p` is.
@pytest.mark.parametrize(("bad_n", "match"), [(-1, "non-negative integer, got -1"), (2**63, "must be at most")])
def test_construct_out_of_range_scalar_n_raises(bad_n: int, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Binomial(n=bad_n, p=0.5)


def test_construct_accepts_a_column_count_past_the_scalar_bound() -> None:
    # The bound is the kwargs wire's, not the distribution's: a column keeps the whole `u64` range.
    got = pl.DataFrame({"n": [2**64 - 1]}, schema={"n": pl.UInt64}).select(r=Binomial(pl.col("n"), 0.5).mean())
    assert got["r"][0] == pytest.approx((2**64 - 1) * 0.5)
