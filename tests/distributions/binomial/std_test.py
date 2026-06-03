from __future__ import annotations

import math

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import Binomial

from .conftest import N_TRIALS


@pytest.mark.parametrize("p", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_std_scalar(p: float, unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=Binomial(N_TRIALS, p).std()).item(0, "v")
    assert result == pytest.approx(math.sqrt(N_TRIALS * p * (1 - p)))


def test_std_propagates_null_in_params(params_with_null: pl.DataFrame) -> None:
    result = params_with_null.select(v=Binomial(pl.col("n"), pl.col("p")).std())["v"]
    expected = pl.Series("v", [math.sqrt(10 * 0.3 * 0.7), None, math.sqrt(8 * 0.8 * 0.2)], dtype=pl.Float64)
    assert_series_equal(result, expected)
