from __future__ import annotations

import math

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import Geometric


@pytest.mark.parametrize("p", [0.25, 0.5, 0.75])
def test_std_is_sqrt_of_variance(p: float, unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=Geometric(p=p).std()).item(0, "v")
    variance = unit_frame.select(v=Geometric(p=p).variance()).item(0, "v")
    assert result == pytest.approx(math.sqrt(variance))


def test_std_does_not_overflow_where_the_variance_route_saturates(unit_frame: pl.DataFrame) -> None:
    # std = sqrt((1-p)/p^2) squares p first: below p ~ 1e-154 the square underflows to 0 and the
    # base-class route returns inf where sqrt(1-p)/p is still finite (~1e200 here).
    p = 1e-200
    result = unit_frame.select(v=Geometric(p=p).std()).item(0, "v")
    assert result == pytest.approx(math.sqrt(1 - p) / p)


def test_std_propagates_null_in_p(p_with_null: pl.DataFrame) -> None:
    result = p_with_null.select(v=Geometric(p=pl.col("p")).std())["v"]
    expected = pl.Series(
        "v",
        [math.sqrt(1 - 0.3) / 0.3, None, math.sqrt(1 - 0.8) / 0.8],
        dtype=pl.Float64,
    )
    assert_series_equal(result, expected)
