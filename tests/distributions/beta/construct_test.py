from __future__ import annotations

import polars as pl
import pytest

from polars_stats import Beta


@pytest.mark.parametrize("bad", [None, True, False, 1, 0, [2.0, 3.0], (2.0,), {"a": 2.0}])
def test_construct_invalid_a_type_raises(bad: object) -> None:
    with pytest.raises(TypeError, match="a should be a float or IntoExprColumn"):
        Beta(a=bad, b=1.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [None, True, False, 1, 0, [1.0], (1.0,), {"b": 1.0}])
def test_construct_invalid_b_type_raises(bad: object) -> None:
    with pytest.raises(TypeError, match="b should be a float or IntoExprColumn"):
        Beta(a=2.0, b=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("a", "b"), [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (1.0, -1e-9), (float("inf"), 1.0), (1.0, float("inf"))]
)
def test_construct_scalar_invalid_shape_defers_to_eval(a: float, b: float) -> None:
    # No early Python validation (matching Normal / Uniform): construction succeeds; the invalid
    # shape surfaces as a ComputeError when a method is evaluated, not a ValueError here.
    Beta(a=a, b=b)
    with pytest.raises(pl.exceptions.ComputeError, match="a and b must be finite and strictly positive"):
        pl.DataFrame({"x": [0.5]}).select(r=Beta(a=a, b=b).pdf(pl.col("x")))


def test_construct_column_params_defers_validation() -> None:
    # A non-positive shape on a column is not knowable at construction; it must not raise here.
    Beta(a=pl.col("a"), b=pl.col("b"))
    Beta(a="a", b="b")
