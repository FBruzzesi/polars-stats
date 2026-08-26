from __future__ import annotations

import math
from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.testing import assert_series_equal, assert_series_not_equal

from polars_stats import Geometric

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize("size", [0, 100, 1_000])
def test_sample_basic_properties(
    size: int,
    frame: Callable[..., pl.DataFrame],
    seed: int,
) -> None:
    result = frame(size=size).lazy().with_columns(geometric=Geometric(p=0.3).sample(seed=seed))
    assert result.collect_schema()["geometric"] == pl.UInt64
    assert result.collect().height == size


@pytest.mark.parametrize("p", [0.1, 0.3, 0.8, pl.col("p1"), pl.col("p2")])
def test_sample_seed_reproducible(
    p: float | pl.Expr,
    frame: Callable[..., pl.DataFrame],
    seed: int,
) -> None:
    dframe = frame()
    s1 = dframe.with_columns(k=Geometric(p=p).sample(seed=seed))["k"]
    s2 = dframe.with_columns(k=Geometric(p=p).sample(seed=seed))["k"]
    assert_series_equal(s1, s2)


@pytest.mark.parametrize("p", [0.1, 0.3, 0.8, pl.col("p1"), pl.col("p2")])
def test_sample_different_seeds_differ(
    p: float | pl.Expr,
    frame: Callable[..., pl.DataFrame],
    seed: int,
) -> None:
    dframe = frame()
    s1 = dframe.with_columns(k=Geometric(p=p).sample(seed=123))["k"]
    s2 = dframe.with_columns(k=Geometric(p=p).sample(seed=seed))["k"]
    assert_series_not_equal(s1, s2)


@pytest.mark.parametrize("p", [0.05, 0.3, 0.8])
def test_sample_support_is_the_positive_integers(p: float, frame: Callable[..., pl.DataFrame]) -> None:
    result = frame().with_columns(k=Geometric(p=p).sample(seed=7))
    assert (result["k"] >= 1).all()


def test_sample_extreme_p_is_always_one(frame: Callable[..., pl.DataFrame]) -> None:
    result = frame().with_columns(k=Geometric(p=1.0).sample(seed=7))
    assert (result["k"] == 1).all()


# `1.0 - p` rounds to exactly `1.0` at or below this, which is what made a `1.0 - p` log base
# collapse and every draw land outside the support. The draw enters through `log1p(-p)` instead.
_ONE_MINUS_P_ROUNDS_TO_ONE = 2.0**-54


@pytest.mark.parametrize("p", [1e-8, 1e-16, _ONE_MINUS_P_ROUNDS_TO_ONE, 1e-17, 1e-19])
def test_sample_support_holds_below_the_one_minus_p_threshold(p: float, frame: Callable[..., pl.DataFrame]) -> None:
    """A small `p` is a valid parameterisation, so its draws must still be positive integers.

    The regime at and below `2 ** -54` is the one a `1.0 - p` log base cannot express: the base is
    exactly `1.0`, its log is `0.0`, and every draw saturates to `0`, below the support, while
    `mean()` correctly reports `1 / p`. Nothing above `p = 0.05` reaches it.
    """
    result = frame().with_columns(k=Geometric(p=p).sample(seed=7))
    assert (result["k"] >= 1).all()


@pytest.mark.parametrize("p", [1e-8, 1e-12, 1e-17])
def test_sample_mean_tracks_inverse_p_into_the_small_p_regime(p: float, frame: Callable[..., pl.DataFrame]) -> None:
    """Being inside the support is not enough: the draws must have the right scale.

    Clamping a broken draw up to the support floor would satisfy the support test above while
    answering `1` where the mean is `1e17`, so this pins the distribution rather than its range.
    """
    n = 100_000
    result = frame(n).with_columns(k=Geometric(p=p).sample(seed=123))
    observed = result["k"].cast(pl.Float64).mean()
    assert isinstance(observed, float)
    assert observed == pytest.approx(1 / p, rel=0.02)


def test_sample_saturates_where_the_trial_count_outgrows_uint64(frame: Callable[..., pl.DataFrame]) -> None:
    """A small enough `p` puts the trial count past `u64::MAX`, where the draw saturates.

    A dtype limit rather than an algorithm one, and the one place a draw is deliberately not the
    inverse transform's answer. A single draw saturates with probability `exp(-u64::MAX * p)`, which
    is why this uses a `p` where that is indistinguishable from 1 rather than one near the onset
    (about 16% of draws at `p = 1e-19`). Pinned so it stays a known boundary.
    """
    result = frame().with_columns(k=Geometric(p=1e-300).sample(seed=7))
    assert (result["k"] == 2**64 - 1).all()


# The largest trial count `UInt64` holds, and the value a saturating cast pins a draw to.
_U64_MAX = 2**64 - 1

# Below this `p` the generator's smallest `u` (about `2 ** -53`) can already reach `u64::MAX`, so
# saturation becomes possible at all: `exp(-u64::MAX * p) >= 2 ** -53` from here down.
_SATURATION_ONSET = 2e-18


@pytest.mark.parametrize("p", [1e-19, 1e-20])
def test_sample_saturated_fraction_matches_the_documented_probability(
    p: float, frame: Callable[..., pl.DataFrame]
) -> None:
    """Through the onset band a draw saturates with probability `exp(-u64::MAX * p)`, not all or nothing.

    That probability is what docs/explanation/accuracy.md quotes, and `p = 1e-300` above only pins
    the end where it is indistinguishable from `1`. Between `_SATURATION_ONSET` and `p = 1e-20` the
    fraction climbs from `0` to 83%, so a change to the cast, to the clamp, or to the uniform's
    distribution would move a boundary neither of the all-or-nothing tests can see.
    """
    n = 200_000
    result = frame(n).with_columns(k=Geometric(p=p).sample(seed=5))
    saturated = (result["k"] == _U64_MAX).sum() / n
    assert saturated == pytest.approx(math.exp(-_U64_MAX * p), rel=0.02)


@pytest.mark.parametrize("p", [1e-17, 10 * _SATURATION_ONSET])
def test_sample_never_saturates_above_the_onset(p: float, frame: Callable[..., pl.DataFrame]) -> None:
    """Above `_SATURATION_ONSET` no `u` the generator can produce reaches `u64::MAX`.

    The other side of the band: `exp(-u64::MAX * p)` drops below the smallest `u`, so saturation
    stops being reachable rather than merely rare. Pins the onset from above, where a wrongly
    documented threshold would otherwise leave a silently biased regime looking safe.
    """
    result = frame(50_000).with_columns(k=Geometric(p=p).sample(seed=5))
    assert (result["k"] < _U64_MAX).all()


@pytest.mark.parametrize("p", [1e-8, _ONE_MINUS_P_ROUNDS_TO_ONE, 1e-17, 1e-19])
def test_sample_fast_path_matches_per_row_in_the_small_p_regime(p: float) -> None:
    """Both sampler paths draw identically where the property suite cannot look.

    `tests/property/_specs.py` floors the geometric `p` at 0.05 so its finite-support sum stays
    tractable, which is why `test_sample_scalar_fast_path_matches_per_row` held while every draw
    below `2 ** -54` was `0`: the two paths were equally wrong. Agreement is only worth something
    alongside the value tests above, and only in the regime that broke.
    """
    frame = pl.DataFrame({"p": [p] * 500})
    fast = frame.with_columns(k=Geometric(p=p).sample(seed=11))["k"]
    per_row = frame.with_columns(k=Geometric(p=pl.col("p")).sample(seed=11))["k"]
    assert_series_equal(fast, per_row)


@pytest.mark.parametrize("p", [1e-8, _ONE_MINUS_P_ROUNDS_TO_ONE, 1e-19])
def test_samples_fast_path_matches_per_row_in_the_small_p_regime(p: float) -> None:
    """The multi-draw twin of the above: one built state per row, `size` draws off it."""
    frame = pl.DataFrame({"p": [p] * 200})
    fast = frame.with_columns(k=Geometric(p=p).samples(size=4, seed=11))["k"]
    per_row = frame.with_columns(k=Geometric(p=pl.col("p")).samples(size=4, seed=11))["k"]
    assert_series_equal(fast, per_row)


@pytest.mark.parametrize("p", [0.05, 0.3, 0.5, 0.8])
def test_sample_mean_close_to_inverse_p_for_large_n(p: float, frame: Callable[..., pl.DataFrame]) -> None:
    n = 100_000
    tolerance = 0.01 * (1 / p)
    result = frame(n).with_columns(k=Geometric(p=p).sample(seed=123))
    observed = result["k"].mean()
    assert isinstance(observed, float)
    assert abs(observed - 1 / p) < tolerance


def test_sample_null_p_row_is_null(seed: int) -> None:
    dframe = pl.DataFrame({"p": [0.5, None, 0.5]}, schema={"p": pl.Float64})
    result = dframe.select(k=Geometric(p=pl.col("p")).sample(seed=seed))["k"]
    assert result[0] is not None
    assert result[1] is None
