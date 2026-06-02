from __future__ import annotations

import numpy as np
import polars as pl
from polars.testing import assert_series_equal

from polars_stats import Uniform


def test_ppf_is_cdf_inverse(bounds: tuple[float, float]) -> None:
    mn, mx = bounds
    interior = [0.1, 0.5, 0.9]
    u = Uniform(min=mn, max=mx)
    out = pl.DataFrame({"q": interior}).select(r=u.cdf(u.ppf(pl.col("q"))))["r"]
    np.testing.assert_allclose(out.to_numpy(), interior, atol=1e-12, rtol=0)


def test_ppf_out_of_range_is_null() -> None:
    df = pl.DataFrame({"q": [-0.1, 0.0, 0.5, 1.0, 1.1]})
    result = df.select(r=Uniform(min=0.0, max=1.0).ppf(pl.col("q")))["r"]
    expected = pl.Series("r", [None, 0.0, 0.5, 1.0, None], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_ppf_propagates_null_in_quantile() -> None:
    df = pl.DataFrame({"q": [0.2, None, 0.8]}, schema={"q": pl.Float64})
    result = df.select(r=Uniform(min=0.0, max=2.0).ppf(pl.col("q")))["r"]
    expected = pl.Series("r", [0.4, None, 1.6], dtype=pl.Float64)
    assert_series_equal(result, expected)
