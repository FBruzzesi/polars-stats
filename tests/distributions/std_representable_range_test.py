"""`std()` must hold wherever the standard deviation is representable, not wherever the variance is.

The base-class `std()` is `variance().sqrt()`, which squares a scale parameter and then unsquares it.
The round trip is lossless in the middle of the range and total at the ends: squaring saturates ~300
decades before the answer does, so `Normal(0, 1e200).std()` returned `inf` and
`Normal(0, 1e-200).std()` returned `0.0`, for a quantity that is exactly the parameter passed in.

`LogNormal.std` was fixed on its own first, which is the mistake this file exists to stop repeating:
the defect belonged to `_base.py::std`, so every distribution whose variance is a squared scale had
it. Oracled by closed forms, since `scipy.stats` composes the same way and returns `inf` too.

One case is deliberately absent. `Beta.variance()` returns `NaN` below shapes of ~`1e-154`
(`a * b` underflows and `(a + b) ** 2` with it, so the ratio is `0 / 0`) against a true `0.5`. That
is the same class but a `variance` defect rather than a `std` one, it predates this work, and fixing
it is a separate change; see the review notes rather than treating this file as covering it.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import polars as pl
import pytest

from polars_stats import Exponential, Geometric, LogNormal, Normal, Uniform

if TYPE_CHECKING:
    from polars_stats.distributions._base import _UnivariateDistribution

# One parameter per decade band that the squaring round trip destroys: `x ** 2` overflows above
# ~1.34e154 and underflows below ~1.49e-162, while `x` itself is representable to 1.8e308 / 5e-324.
_EXTREME_SCALES = [1e-300, 1e-200, 1e-170, 1e170, 1e200, 1e300]


@pytest.mark.parametrize("sigma", _EXTREME_SCALES, ids=lambda s: f"sigma={s:.0e}")
def test_normal_std_is_sigma_at_any_representable_scale(sigma: float) -> None:
    """`Normal(mu, sigma).std()` is `sigma` exactly: there is nothing to compute."""
    got = pl.DataFrame({"z": [0.0]}).select(r=Normal(mu=0.0, sigma=sigma).std())["r"].item()
    assert got == sigma


@pytest.mark.parametrize("rate", _EXTREME_SCALES, ids=lambda r: f"rate={r:.0e}")
def test_exponential_std_is_the_reciprocal_rate(rate: float) -> None:
    """`Exponential(rate).std()` is `1 / rate`, the same expression as its `mean`."""
    got = pl.DataFrame({"z": [0.0]}).select(r=Exponential(rate=rate).std())["r"].item()
    assert got == pytest.approx(1.0 / rate, rel=1e-15, abs=0.0)


@pytest.mark.parametrize("span", _EXTREME_SCALES, ids=lambda s: f"span={s:.0e}")
def test_uniform_std_is_the_span_over_root_twelve(span: float) -> None:
    """`Uniform(0, span).std()` is `span / sqrt(12)` across the full representable range."""
    got = pl.DataFrame({"z": [0.0]}).select(r=Uniform(min=0.0, max=span).std())["r"].item()
    assert got == pytest.approx(span / math.sqrt(12.0), rel=1e-15, abs=0.0)


# `p` is a probability, so only the underflow half of `_EXTREME_SCALES` is reachable here: `p ** 2`
# underflows below ~1.49e-162 while `sqrt(1 - p) / p` stays finite down to ~5.6e-309, where `1 / p`
# itself overflows.
_EXTREME_PROBABILITIES = [s for s in _EXTREME_SCALES if s < 1.0]


@pytest.mark.parametrize("p", _EXTREME_PROBABILITIES, ids=lambda p: f"p={p:.0e}")
def test_geometric_std_is_sqrt_one_minus_p_over_p(p: float) -> None:
    """`Geometric(p).std()` is `sqrt(1 - p) / p` where the squared-variance route returns `inf`."""
    got = pl.DataFrame({"z": [0.0]}).select(r=Geometric(p=p).std())["r"].item()
    assert math.isfinite(got)
    assert got == pytest.approx(math.sqrt(1 - p) / p, rel=1e-15, abs=0.0)


@pytest.mark.parametrize("sigma", [1.0, 18.0, 20.0, 26.0], ids=lambda s: f"sigma={s}")
def test_lognormal_std_outlives_its_variance(sigma: float) -> None:
    """`LogNormal.std()` stays finite past `sigma ~ 18.8`, where the variance genuinely overflows.

    Unlike the three above, `inf` really is the right answer for `variance()` here, which is why the
    fix had to be a separate expression rather than a rearrangement of the same one.
    """
    frame = pl.DataFrame({"z": [0.0]})
    got = frame.select(r=LogNormal(mu=0.0, sigma=sigma).std())["r"].item()
    expected = math.exp(0.5 * math.log(math.expm1(sigma**2)) + sigma**2 / 2)
    assert math.isfinite(got)
    assert got == pytest.approx(expected, rel=1e-13, abs=0.0)


@pytest.mark.parametrize(
    ("dist", "expected"),
    [
        (Normal(mu=0.0, sigma=2.0), 2.0),
        (Exponential(rate=4.0), 0.25),
        (Uniform(min=-3.0, max=7.0), 10.0 / math.sqrt(12.0)),
        (Geometric(p=0.25), math.sqrt(0.75) / 0.25),
    ],
    ids=["normal", "exponential", "uniform", "geometric"],
)
def test_std_squared_still_agrees_with_variance_in_the_ordinary_range(
    dist: _UnivariateDistribution, expected: float
) -> None:
    """The overrides must not drift from `variance()` where both are representable."""
    frame = pl.DataFrame({"z": [0.0]})
    got = frame.select(s=dist.std(), v=dist.variance())
    assert got["s"].item() == pytest.approx(expected, rel=1e-15, abs=0.0)
    assert got["s"].item() ** 2 == pytest.approx(got["v"].item(), rel=1e-14, abs=0.0)
