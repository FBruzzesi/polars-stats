from __future__ import annotations

import polars as pl
import pytest

from polars_stats import Beta
from tests._polars_compat import assert_series_equal


@pytest.mark.parametrize("shape", [0.5, 1.0, 2.0, 5.0])
def test_ppf_at_half_is_centre_for_symmetric_shapes(shape: float) -> None:
    result = pl.DataFrame({"q": [0.5]}).select(r=Beta(a=shape, b=shape).ppf(pl.col("q")))["r"].item()
    assert result == pytest.approx(0.5, abs=1e-9)


def test_ppf_is_cdf_inverse(params: tuple[float, float]) -> None:
    a, b = params
    interior = [0.05, 0.25, 0.5, 0.75, 0.95]
    dist = Beta(a=a, b=b)
    result = pl.DataFrame({"q": interior}).select(r=dist.cdf(dist.ppf(pl.col("q"))))["r"]
    expected = pl.Series("r", interior, dtype=pl.Float64)
    assert_series_equal(result, expected, rel_tol=0.0, abs_tol=1e-9)


def test_ppf_endpoints_are_support_bounds() -> None:
    df = pl.DataFrame({"q": [0.0, 1.0]})
    result = df.select(r=Beta(a=2.0, b=3.0).ppf(pl.col("q")))["r"]
    expected = pl.Series("r", [0.0, 1.0], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_ppf_out_of_range_is_null() -> None:
    df = pl.DataFrame({"q": [-0.1, 0.0, 0.5, 1.0, 1.1]})
    result = df.select(r=Beta(a=2.0, b=2.0).ppf(pl.col("q")))["r"]
    expected = pl.Series("r", [None, 0.0, 0.5, 1.0, None], dtype=pl.Float64)
    assert_series_equal(result, expected, rel_tol=0.0, abs_tol=1e-9)


def test_ppf_propagates_null_in_quantile() -> None:
    df = pl.DataFrame({"q": [0.5, None]}, schema={"q": pl.Float64})
    result = df.select(r=Beta(a=2.0, b=2.0).ppf(pl.col("q")))["r"]
    expected = pl.Series("r", [0.5, None], dtype=pl.Float64)
    assert_series_equal(result, expected, rel_tol=0.0, abs_tol=1e-9)
