"""`Beta(1, 1)` is `Uniform(0, 1)`; every closed-form method must agree across the two."""

from __future__ import annotations

import polars as pl
import pytest

from polars_stats import Beta, Uniform
from tests._polars_compat import assert_series_equal

_XS = [-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5]  # interior, both boundaries, and both sides outside
_QS = [0.0, 0.1, 0.5, 0.9, 1.0]

_BETA = Beta(a=1.0, b=1.0)
_UNIFORM = Uniform(min=0.0, max=1.0)


@pytest.mark.parametrize("method", ["pdf", "log_pdf", "cdf", "log_cdf", "sf", "log_sf"])
def test_value_keyed_methods_match_uniform(method: str) -> None:
    df = pl.DataFrame({"x": _XS})
    via_beta = df.select(r=getattr(_BETA, method)(pl.col("x")))["r"]
    via_uniform = df.select(r=getattr(_UNIFORM, method)(pl.col("x")))["r"]
    assert_series_equal(via_beta, via_uniform, rel_tol=0.0, abs_tol=1e-12)


@pytest.mark.parametrize("method", ["ppf", "isf"])
def test_quantile_methods_match_uniform(method: str) -> None:
    # The inverse regularized incomplete beta is Newton-refined, so agreement is close but not exact.
    df = pl.DataFrame({"q": _QS})
    via_beta = df.select(r=getattr(_BETA, method)(pl.col("q")))["r"]
    via_uniform = df.select(r=getattr(_UNIFORM, method)(pl.col("q")))["r"]
    assert_series_equal(via_beta, via_uniform, rel_tol=0.0, abs_tol=1e-9)


@pytest.mark.parametrize("method", ["mean", "variance", "std", "median", "entropy"])
def test_scalar_methods_match_uniform(method: str) -> None:
    frame = pl.DataFrame({"_": [0]})
    via_beta = frame.select(r=getattr(_BETA, method)()).item(0, "r")
    via_uniform = frame.select(r=getattr(_UNIFORM, method)()).item(0, "r")
    assert via_beta == pytest.approx(via_uniform, abs=1e-9)
