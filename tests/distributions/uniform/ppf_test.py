from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_series_equal

from polars_stats import Uniform

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

PARAMS = [(0.0, 1.0), (-2.0, 3.0), (2.0, 5.0), (-5.0, -1.0), (0.0, 1e-3)]
_QUANTILES = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]


@pytest.mark.parametrize(("mn", "mx"), PARAMS)
def test_ppf_matches_scipy(mn: float, mx: float, scipy_uniform: Callable[[float, float], Any]) -> None:
    got = pl.DataFrame({"q": _QUANTILES}).select(r=Uniform(min=mn, max=mx).ppf(pl.col("q")))["r"].to_numpy()
    np.testing.assert_allclose(got, scipy_uniform(mn, mx).ppf(_QUANTILES), atol=1e-12, rtol=0)


@pytest.mark.parametrize(("mn", "mx"), PARAMS)
def test_ppf_is_cdf_inverse(mn: float, mx: float) -> None:
    interior = [0.1, 0.5, 0.9]
    u = Uniform(min=mn, max=mx)
    out = pl.DataFrame({"q": interior}).select(r=u.cdf(u.ppf(pl.col("q"))))["r"]
    np.testing.assert_allclose(out.to_numpy(), interior, atol=1e-12, rtol=0)


def test_ppf_out_of_range_is_null() -> None:
    df = pl.DataFrame({"q": [-0.1, 0.0, 0.5, 1.0, 1.1]})
    got = df.select(r=Uniform(min=0.0, max=1.0).ppf(pl.col("q")))["r"]
    expected = pl.Series("r", [None, 0.0, 0.5, 1.0, None], dtype=pl.Float64)
    assert_series_equal(got, expected)


def test_ppf_propagates_null_in_quantile() -> None:
    df = pl.DataFrame({"q": [0.2, None, 0.8]}, schema={"q": pl.Float64})
    got = df.select(r=Uniform(min=0.0, max=2.0).ppf(pl.col("q")))["r"]
    expected = pl.Series("r", [0.4, None, 1.6], dtype=pl.Float64)
    assert_series_equal(got, expected)
