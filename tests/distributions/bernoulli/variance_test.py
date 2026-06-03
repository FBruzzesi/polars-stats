from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import Bernoulli


@pytest.mark.parametrize("p", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_variance_scalar(p: float, unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=Bernoulli(p=p).variance()).item(0, "v")
    assert result == pytest.approx(p * (1 - p))


def test_variance_column_p() -> None:
    probs = [0.1, 0.4, 0.6, 0.9]
    df = pl.DataFrame({"p": probs})
    result = df.select(v=Bernoulli(p=pl.col("p")).variance())["v"]
    expected = pl.Series("v", [x * (1 - x) for x in probs], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_variance_propagates_null_in_p(p_with_null: pl.DataFrame) -> None:
    result = p_with_null.select(v=Bernoulli(p=pl.col("p")).variance())["v"]
    expected = pl.Series("v", [0.3 * 0.7, None, 0.8 * 0.2], dtype=pl.Float64)
    assert_series_equal(result, expected)
