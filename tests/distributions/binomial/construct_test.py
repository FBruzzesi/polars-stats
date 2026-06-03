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


@pytest.mark.parametrize("n", [0, 1, 10, "n", pl.col("n"), pl.Series("n", [5])])
def test_construct_accepts_valid_n_types(n: object) -> None:
    # int and IntoExprColumn are accepted; value validation is deferred to evaluation.
    Binomial(n=n, p=0.5)  # type: ignore[arg-type]
