from __future__ import annotations

import math

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import Bernoulli


@pytest.mark.parametrize(
    ("p", "quantile", "expected"),
    [
        (0.3, 0.0, 0.0),
        (0.3, 0.5, 0.0),  # 0.5 <= 1 - 0.3 = 0.7
        (0.3, 0.7, 0.0),  # boundary: q <= 1 - p → 0
        (0.3, 0.8, 1.0),
        (0.3, 1.0, 1.0),
        (0.0, 0.5, 0.0),
        (1.0, 0.5, 1.0),
        (1.0, 0.0, 0.0),  # 0 > 1 - 1 = 0 is False
    ],
)
def test_ppf_scalar(p: float, quantile: float, expected: float, unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=Bernoulli(p=p).ppf(quantile)).item(0, "v")
    assert result == expected


def test_ppf_propagates_null_in_quantile() -> None:
    df = pl.DataFrame({"q": [0.1, None, 0.9]}, schema={"q": pl.Float64})
    result = df.select(v=Bernoulli(p=0.3).ppf(pl.col("q")))["v"]
    expected = pl.Series("v", [0.0, None, 1.0], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_ppf_propagates_nan_in_quantile(unit_frame: pl.DataFrame) -> None:
    # A NaN quantile propagates as NaN (scipy semantics) rather than falling through
    # `NaN > 1 - p` (true under polars' NaN-greatest ordering) to a confident 1.0.
    result = unit_frame.select(v=Bernoulli(p=0.3).ppf(float("nan"))).item(0, "v")
    assert math.isnan(result)


def test_ppf_propagates_null_in_p(p_with_null: pl.DataFrame) -> None:
    # ppf(0.5) is 0.0 when p <= 0.5, 1.0 when p > 0.5, null when p is null.
    result = p_with_null.select(v=Bernoulli(p=pl.col("p")).ppf(0.5))["v"]
    expected = pl.Series("v", [0.0, None, 1.0], dtype=pl.Float64)
    assert_series_equal(result, expected)
