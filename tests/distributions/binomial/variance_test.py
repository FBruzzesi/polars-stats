from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import Binomial
from tests.distributions.binomial.conftest import N_TRIALS


@pytest.mark.parametrize("p", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_variance_scalar(p: float, unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=Binomial(N_TRIALS, p).variance()).item(0, "v")
    assert result == pytest.approx(N_TRIALS * p * (1 - p))


def test_variance_column_params() -> None:
    df = pl.DataFrame({"n": [5, 10, 20], "p": [0.1, 0.4, 0.6]})
    result = df.select(v=Binomial(pl.col("n"), pl.col("p")).variance())["v"]
    expected = pl.Series("v", [nn * pp * (1 - pp) for nn, pp in [(5, 0.1), (10, 0.4), (20, 0.6)]], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_variance_propagates_null_in_params(params_with_null: pl.DataFrame) -> None:
    result = params_with_null.select(v=Binomial(pl.col("n"), pl.col("p")).variance())["v"]
    expected = pl.Series("v", [10 * 0.3 * 0.7, None, 8 * 0.8 * 0.2], dtype=pl.Float64)
    assert_series_equal(result, expected)
