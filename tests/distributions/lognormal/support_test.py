from __future__ import annotations

import polars as pl
from polars.testing import assert_series_equal

from polars_stats import LogNormal

# The LogNormal support is `x > 0`. For `x <= 0` the density and cdf are 0 and the survival function
# is 1, matching `scipy.stats.lognorm`. The boundary `x == 0` is included in the "outside" branch.


def test_pdf_is_zero_at_or_below_zero() -> None:
    df = pl.DataFrame({"x": [-2.0, -1e-9, 0.0]})
    result = df.select(r=LogNormal().pdf(pl.col("x")))["r"]
    assert_series_equal(result, pl.Series("r", [0.0, 0.0, 0.0], dtype=pl.Float64))


def test_cdf_is_zero_at_or_below_zero() -> None:
    df = pl.DataFrame({"x": [-2.0, -1e-9, 0.0]})
    result = df.select(r=LogNormal().cdf(pl.col("x")))["r"]
    assert_series_equal(result, pl.Series("r", [0.0, 0.0, 0.0], dtype=pl.Float64))


def test_sf_is_one_at_or_below_zero() -> None:
    df = pl.DataFrame({"x": [-2.0, -1e-9, 0.0]})
    result = df.select(r=LogNormal().sf(pl.col("x")))["r"]
    assert_series_equal(result, pl.Series("r", [1.0, 1.0, 1.0], dtype=pl.Float64))


def test_log_pdf_is_neg_inf_at_or_below_zero() -> None:
    df = pl.DataFrame({"x": [-2.0, 0.0]})
    result = df.select(r=LogNormal().log_pdf(pl.col("x")))["r"]
    assert_series_equal(result, pl.Series("r", [float("-inf"), float("-inf")], dtype=pl.Float64))
