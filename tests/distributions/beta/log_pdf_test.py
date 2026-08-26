from __future__ import annotations

import math

import polars as pl

from polars_stats import Beta
from tests._polars_compat import assert_series_equal


def test_log_pdf_equals_log_of_pdf(value_grid: list[float]) -> None:
    b = Beta(a=2.0, b=3.0)
    result = pl.DataFrame({"x": value_grid}).select(diff=b.log_pdf(pl.col("x")) - b.pdf(pl.col("x")).log())["diff"]
    expected = pl.Series("diff", [0.0] * result.len(), dtype=pl.Float64)
    assert_series_equal(result, expected, rel_tol=0.0, abs_tol=1e-12)


def test_log_pdf_finite_in_the_upper_1e9_band() -> None:
    # The log-density is finite right up to the endpoint (positive for small shapes), so no value
    # below 1 may return -inf. statrs 0.19.0 collapsed this whole band to -inf by comparing the
    # evaluation point to 1.0 with an absolute epsilon; 0.19.1 restored the exact comparison. The
    # expectation is the closed form rather than a recorded number, so this stays a live check.
    x = 1.0 - 1e-9
    df = pl.DataFrame({"x": [x]})
    for a, b in [(2.0, 3.0), (0.05, 0.05), (5.0, 1e-3)]:
        ln_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        expected_val = (a - 1.0) * math.log(x) + (b - 1.0) * math.log(1.0 - x) - ln_beta
        result = df.select(r=Beta(a=a, b=b).log_pdf(pl.col("x")))["r"]
        expected = pl.Series("r", [expected_val], dtype=pl.Float64)
        assert_series_equal(result, expected, rel_tol=1e-13, abs_tol=0.0)


def test_log_pdf_outside_support_is_neg_inf() -> None:
    df = pl.DataFrame({"x": [-0.5, 1.5]})
    result = df.select(r=Beta(a=2.0, b=3.0).log_pdf(pl.col("x")))["r"]
    expected = pl.Series("r", [float("-inf"), float("-inf")], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_log_pdf_propagates_null_in_value() -> None:
    df = pl.DataFrame({"x": [0.5, None]}, schema={"x": pl.Float64})
    result = df.select(r=Beta(a=2.0, b=2.0).log_pdf(pl.col("x")))["r"]
    expected = pl.Series("r", [math.log(1.5), None], dtype=pl.Float64)
    assert_series_equal(result, expected)
