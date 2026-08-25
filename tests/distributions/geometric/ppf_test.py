from __future__ import annotations

import math

import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import Geometric


@pytest.mark.parametrize(
    ("p", "quantile", "expected"),
    [
        (0.3, 0.0, 1.0),  # smallest support point, mapped explicitly at the degenerate endpoint
        (0.3, 1e-300, 1.0),  # every representable q > 0 is >= cdf(1)
        (0.3, 0.29, 1.0),  # 0.29 <= cdf(1) = 0.3
        (0.3, 0.31, 2.0),  # boundary crossed between k=1 and k=2
        (0.3, 0.5, 2.0),
        (0.3, 0.51, 3.0),  # fl(0.51) > cdf(2) = 0.51: the stored quantile already crosses k = 2
        (0.5, 0.75, 2.0),  # exact power-of-two boundary: cdf(2) = 0.75 lands on the quantile bit-for-bit
        (0.3, 0.52, 3.0),
        (0.3, 1.0, float("inf")),  # unbounded support: only the limit answers q = 1
        (0.5, 0.5, 1.0),
        (1.0, 0.5, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
    ],
)
def test_ppf_scalar(p: float, quantile: float, expected: float, unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=Geometric(p=p).ppf(quantile)).item(0, "v")
    assert result == expected


def test_ppf_matches_closed_form_for_small_p(unit_frame: pl.DataFrame) -> None:
    # Interior-quantile check in a regime where a naive log(1 - q) / log(1 - p) loses
    # digits on both sides.
    p, q = 1e-13, 1e-11
    expected = math.ceil(math.log1p(-q) / math.log1p(-p))
    got = unit_frame.select(v=Geometric(p=p).ppf(q)).item(0, "v")
    assert got == expected


@pytest.mark.parametrize("quantile", [-0.1, 1.5])
def test_ppf_out_of_range_quantile_is_null(quantile: float, unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=Geometric(p=0.3).ppf(quantile)).item(0, "v")
    assert result is None


def test_ppf_propagates_nan_in_quantile(unit_frame: pl.DataFrame) -> None:
    result = unit_frame.select(v=Geometric(p=0.3).ppf(float("nan"))).item(0, "v")
    assert math.isnan(result)


def test_ppf_propagates_null_in_quantile() -> None:
    df = pl.DataFrame({"q": [0.1, None, 0.9]}, schema={"q": pl.Float64})
    result = df.select(v=Geometric(p=0.3).ppf(pl.col("q")))["v"]
    expected = pl.Series("v", [1.0, None, 7.0], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_ppf_propagates_null_in_p(p_with_null: pl.DataFrame) -> None:
    result = p_with_null.select(v=Geometric(p=pl.col("p")).ppf(0.5))["v"]
    expected = pl.Series("v", [2.0, None, 1.0], dtype=pl.Float64)
    assert_series_equal(result, expected)
