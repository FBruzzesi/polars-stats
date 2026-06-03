from __future__ import annotations

import polars as pl
import pytest

from polars_stats import LogNormal


@pytest.mark.parametrize("bad", [None, True, False, 1, 0, [0.0, 1.0], (0.0,), {"mu": 0.0}])
def test_construct_invalid_mu_type_raises(bad: object) -> None:
    with pytest.raises(TypeError, match="mu should be a float or IntoExprColumn"):
        LogNormal(mu=bad, sigma=1.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [None, True, False, 1, 0, [1.0], (1.0,), {"sigma": 1.0}])
def test_construct_invalid_sigma_type_raises(bad: object) -> None:
    with pytest.raises(TypeError, match="sigma should be a float or IntoExprColumn"):
        LogNormal(mu=0.0, sigma=bad)  # type: ignore[arg-type]


def test_construct_defaults_to_standard_lognormal() -> None:
    # Default mu=0, sigma=1: median is exp(0) = 1, so cdf(1) = 0.5.
    result = pl.DataFrame({"x": [1.0]}).select(r=LogNormal().cdf(pl.col("x")))["r"].item()
    assert result == pytest.approx(0.5, abs=1e-12)


@pytest.mark.parametrize("bad_sigma", [0.0, -1.0, -1e-9])
def test_construct_scalar_non_positive_sigma_defers_to_eval(bad_sigma: float) -> None:
    # No early Python validation (matching Bernoulli / Uniform / Normal): construction succeeds; the
    # invalid scale surfaces as a ComputeError when a method is evaluated, not a ValueError here.
    LogNormal(mu=0.0, sigma=bad_sigma)
    with pytest.raises(pl.exceptions.ComputeError, match="sigma must be finite and strictly positive"):
        pl.DataFrame({"x": [0.5]}).select(r=LogNormal(mu=0.0, sigma=bad_sigma).pdf(pl.col("x")))


def test_construct_column_params_defers_validation() -> None:
    # A non-positive sigma on a column is not knowable at construction; it must not raise here.
    LogNormal(mu=pl.col("mu"), sigma=pl.col("sigma"))
    LogNormal(mu="mu", sigma="sigma")
