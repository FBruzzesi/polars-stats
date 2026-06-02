from __future__ import annotations

import polars as pl
import pytest

from polars_stats import Normal


@pytest.mark.parametrize("bad", [None, True, False, 1, 0, [0.0, 1.0], (0.0,), {"mean": 0.0}])
def test_construct_invalid_mean_type_raises(bad: object) -> None:
    with pytest.raises(TypeError, match="mean should be a float or IntoExprColumn"):
        Normal(mean=bad, std_dev=1.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [None, True, False, 1, 0, [1.0], (1.0,), {"std_dev": 1.0}])
def test_construct_invalid_std_dev_type_raises(bad: object) -> None:
    with pytest.raises(TypeError, match="std_dev should be a float or IntoExprColumn"):
        Normal(mean=0.0, std_dev=bad)  # type: ignore[arg-type]


def test_construct_defaults_to_standard_normal() -> None:
    # No-op standard normal: mean=0, std_dev=1 is the default, handled by the same path as any other.
    result = pl.DataFrame({"x": [0.0]}).select(r=Normal().cdf(pl.col("x")))["r"].item()
    assert result == pytest.approx(0.5, abs=1e-12)


@pytest.mark.parametrize("bad_std", [0.0, -1.0, -1e-9])
def test_construct_scalar_non_positive_std_defers_to_eval(bad_std: float) -> None:
    # No early Python validation (matching Bernoulli / Uniform): construction succeeds; the invalid
    # scale surfaces as a ComputeError when a method is evaluated, not a ValueError here.
    Normal(mean=0.0, std_dev=bad_std)
    with pytest.raises(pl.exceptions.ComputeError, match="std_dev must be finite and strictly positive"):
        pl.DataFrame({"x": [0.5]}).select(r=Normal(mean=0.0, std_dev=bad_std).pdf(pl.col("x")))


def test_construct_column_params_defers_validation() -> None:
    # A non-positive std_dev on a column is not knowable at construction; it must not raise here.
    Normal(mean=pl.col("mu"), std_dev=pl.col("sigma"))
    Normal(mean="mu", std_dev="sigma")
