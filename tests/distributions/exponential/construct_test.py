from __future__ import annotations

import polars as pl
import pytest

from polars_stats import Exponential


@pytest.mark.parametrize("bad", [None, True, False, 1, 0, [1.0], (1.0,), {"rate": 1.0}])
def test_construct_invalid_rate_type_raises(bad: object) -> None:
    # `rate` is a float (or IntoExprColumn): an `int` / `bool` is rejected by type, like every
    # float parameter (`coerce_param`).
    with pytest.raises(TypeError, match="rate should be a float or IntoExprColumn"):
        Exponential(rate=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_rate", [0.0, -1.0, -2.5, float("nan")])
def test_construct_scalar_invalid_rate_defers_to_eval(bad_rate: float) -> None:
    # No early Python validation: construction succeeds (matching Uniform / Bernoulli's deferral);
    # the invalid rate surfaces as a ComputeError when a method is evaluated, not a ValueError here.
    Exponential(rate=bad_rate)
    with pytest.raises(pl.exceptions.ComputeError, match="rate must be strictly positive"):
        pl.DataFrame({"x": [0.5]}).select(r=Exponential(rate=bad_rate).pdf(pl.col("x")))


def test_construct_column_rate_defers_validation() -> None:
    # A non-positive rate on a column is not knowable at construction; it must not raise here.
    Exponential(rate=pl.col("rate"))
    Exponential(rate="rate")
