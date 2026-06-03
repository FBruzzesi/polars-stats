from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import Normal


def test_ppf_at_half_is_mean(params: tuple[float, float]) -> None:
    mean, std = params
    result = pl.DataFrame({"q": [0.5]}).select(r=Normal(mean=mean, std_dev=std).ppf(pl.col("q")))["r"].item()
    assert result == pytest.approx(mean, abs=1e-12)


def test_ppf_is_cdf_inverse(params: tuple[float, float]) -> None:
    mean, std = params
    interior = [0.05, 0.25, 0.5, 0.75, 0.95]
    n = Normal(mean=mean, std_dev=std)
    out = pl.DataFrame({"q": interior}).select(r=n.cdf(n.ppf(pl.col("q"))))["r"]
    np.testing.assert_allclose(out.to_numpy(), interior, atol=1e-9, rtol=0)


def test_ppf_endpoints_are_infinite() -> None:
    df = pl.DataFrame({"q": [0.0, 1.0]})
    result = df.select(r=Normal().ppf(pl.col("q")))["r"]
    expected = pl.Series("r", [float("-inf"), float("inf")], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_ppf_out_of_range_is_null() -> None:
    df = pl.DataFrame({"q": [-0.1, 0.0, 0.5, 1.0, 1.1]})
    result = df.select(r=Normal().ppf(pl.col("q")))["r"]
    expected = pl.Series("r", [None, float("-inf"), 0.0, float("inf"), None], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_ppf_propagates_null_in_quantile() -> None:
    df = pl.DataFrame({"q": [0.5, None]}, schema={"q": pl.Float64})
    result = df.select(r=Normal(mean=2.0, std_dev=1.0).ppf(pl.col("q")))["r"]
    expected = pl.Series("r", [2.0, None], dtype=pl.Float64)
    assert_series_equal(result, expected)
