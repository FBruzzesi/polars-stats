from __future__ import annotations

import polars as pl
import pytest

from polars_stats import Uniform


@pytest.mark.parametrize("bad", [None, True, False, 1, 0, [0.0, 1.0], (0.0,), {"min": 0.0}])
def test_construct_invalid_min_type_raises(bad: object) -> None:
    with pytest.raises(TypeError, match="min should be a float or IntoExprColumn"):
        Uniform(min=bad, max=1.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [None, True, False, 1, 0, [0.0, 1.0], (0.0,), {"max": 1.0}])
def test_construct_invalid_max_type_raises(bad: object) -> None:
    with pytest.raises(TypeError, match="max should be a float or IntoExprColumn"):
        Uniform(min=0.0, max=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize(("mn", "mx"), [(1.0, 1.0), (2.0, 1.0), (0.0, -1.0)])
def test_construct_scalar_max_le_min_defers_to_eval(mn: float, mx: float) -> None:
    # No early Python validation: construction succeeds (matching Bernoulli's deferral); the invalid
    # parameterisation surfaces as a ComputeError when a method is evaluated, not a ValueError here.
    Uniform(min=mn, max=mx)
    with pytest.raises(pl.exceptions.ComputeError, match="max must be strictly greater than min"):
        pl.DataFrame({"x": [0.5]}).select(r=Uniform(min=mn, max=mx).pdf(pl.col("x")))


def test_construct_column_bounds_defers_validation() -> None:
    # max <= min on a column is not knowable at construction; it must not raise here.
    Uniform(min=pl.col("lo"), max=pl.col("hi"))
    Uniform(min="lo", max="hi")
