"""`sample`: one draw per row, and what the draw is contracted to be.

`tests/property/sample_test.py` owns reproducibility, the constant-vs-per-row bit equality and the
null-row layering under `hypothesis`. This file owns the claims a fixed frame states better: the
output's shape and dtype, that different seeds and no seed differ, that rows do not alias, that draws
land inside the support, and that the empirical mean tracks the distribution's own.

Invalid and null parameters are `validation_test.py`'s; broadcasting and partition contexts are
`broadcast_test.py`'s.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_series_equal, assert_series_not_equal

from polars_stats import Bernoulli, DiscreteUniform, Geometric, Uniform
from tests._registry import DRIVER_REGIMES, SPECS_BY_NAME, min_max, moment_is_undefined

if TYPE_CHECKING:
    from tests._registry import DistSpec, Regime

_SEED = 42
_CLT_ROWS = 20_000
_CLT_SIGMAS = 6.0
"""How many standard errors of slack the mean check allows.

Six standard errors is far outside what any parameter mistake lands in, and the seed is fixed, so the
check is deterministic rather than flaky-at-a-known-rate.
"""

_PER_GROUP, _GROUPS = 200, 20

_GROUP_SCALE = 0.5 + pl.col("g").cast(pl.Float64) / (2.0 * (_GROUPS - 1))
"""Per-group multiplier in `[0.5, 1.0]`, applied to every parameter at once.

One rule for all nine rows rather than a per-distribution perturbation: halving every parameter
together stays inside each domain (a scaled `p` stays in `[0, 1]`, a scaled scale stays positive,
and scaled bounds keep their ordering), so no row needs a special case.
"""


def _frame(rows: int) -> pl.DataFrame:
    return pl.DataFrame({"i": range(rows)})


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
@pytest.mark.parametrize("rows", [0, 1, 1_000])
def test_a_draw_is_full_length_and_the_recorded_dtype(spec: DistSpec, regime: Regime, rows: int) -> None:
    """Including the empty frame, where a length-1 broadcast would give one row instead of none."""
    out = _frame(rows).lazy().with_columns(z=spec.build(regime).sample(seed=_SEED))
    assert out.collect_schema()["z"] == spec.sample_dtype
    assert out.collect().height == rows


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
def test_different_seeds_differ(spec: DistSpec, regime: Regime) -> None:
    frame = _frame(512)
    first = frame.select(z=spec.build(regime).sample(seed=123))["z"]
    second = frame.select(z=spec.build(regime).sample(seed=_SEED))["z"]
    assert_series_not_equal(first, second)


def test_an_unseeded_draw_varies(spec: DistSpec) -> None:
    """`seed=None` takes OS entropy, so two calls in one process must not agree."""
    frame = _frame(512)
    first = frame.select(z=spec.build("scalar").sample(seed=None))["z"]
    second = frame.select(z=spec.build("scalar").sample(seed=None))["z"]
    assert_series_not_equal(first, second)


def test_rows_do_not_alias(spec: DistSpec) -> None:
    """Each row derives its stream from `(seed, row index)`, so no two rows share a draw.

    Full uniqueness among `Float64` draws is the observable form of that guarantee, so this runs on
    the continuous rows only: a discrete support repeats by construction.
    """
    if not spec.continuous:
        pytest.skip(f"{spec.name} is discrete: its support repeats by construction")

    rows = 50_000
    drawn = _frame(rows).select(z=spec.build("scalar").sample(seed=_SEED))["z"]
    assert drawn.n_unique() == rows


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
def test_draws_lie_inside_the_support(spec: DistSpec, regime: Regime) -> None:
    """Every draw is within `spec.bounds`, which is what a swapped or misread bound breaks."""
    lo, hi = spec.bounds
    low, high = min_max(_frame(4_000).select(z=spec.build(regime).sample(seed=_SEED))["z"])
    assert low >= lo, f"{spec.name} drew {low} below {lo}"
    assert high <= hi, f"{spec.name} drew {high} above {hi}"


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
def test_the_sample_mean_tracks_the_distribution_mean(spec: DistSpec, regime: Regime) -> None:
    """The empirical mean sits within `_CLT_SIGMAS` standard errors of `mean()`.

    The one thing no parity comparison can see: `build_dist` handing statrs its parameters in the
    wrong order still matches scipy on every closed form, because the closed forms never go through
    it. The distribution's own `mean` and `std` are the oracle here, and they are independently
    checked against scipy by `tests/scipy_parity`.

    Nine of these replace nineteen per-distribution moment checks at 200,000 rows, which is a
    duplication cut rather than a speed one: those cost 0.07s of a 12s suite.

    A distribution with no mean has nothing to track;
    `test_a_cauchy_draw_tracks_loc_and_scale_through_its_quartiles` is its parameter-order check instead.
    """
    if moment_is_undefined(spec.name, "mean"):
        pytest.skip(f"{spec.name} has no mean for the draws to track")
    dist = spec.build(regime)
    frame = _frame(_CLT_ROWS)
    moments = frame.select(mean=dist.mean(), std=dist.std())
    expected, spread = moments["mean"][0], moments["std"][0]

    observed = frame.select(z=dist.sample(seed=_SEED))["z"].cast(pl.Float64).mean()
    assert isinstance(observed, float)

    tolerance = _CLT_SIGMAS * spread / math.sqrt(_CLT_ROWS)
    assert abs(observed - expected) <= tolerance, (
        f"{spec.name}: sample mean {observed} is more than {_CLT_SIGMAS} standard errors ({tolerance}) from {expected}"
    )


def test_group_by_draws_per_group(spec: DistSpec) -> None:
    """Every group sees the same row indices `0..n-1`, so constant parameters give identical groups.

    That is the observable form of the per-row seeding contract under an aggregation: a sampler
    seeded from a global position instead of a partition-local one would make the groups differ.
    """
    frame = pl.DataFrame({"g": np.repeat(np.arange(_GROUPS), _PER_GROUP)})
    agg = (
        frame.group_by("g", maintain_order=True)
        .agg(total=spec.build("scalar").sample(seed=_SEED).cast(pl.Float64).sum(), size=pl.len())
        .sort("g")
    )
    assert agg["size"].eq(_PER_GROUP).all()
    assert agg["total"].n_unique() == 1


def test_group_by_with_a_varying_parameter_draws_per_group(spec: DistSpec) -> None:
    """With a parameter that differs by group, the group totals must differ.

    Every group draws the same uniforms (same seed, same row indices `0..n-1`), so the totals can
    only differ through the parameter. A sampler that ignored its parameter column, or read it at a
    partition-local index that collapses to one value, totals identically and fails here. The
    constant-parameter half above cannot see that, since there the two reads are the same value.
    This is the claim the deleted `binomial/sample_test.py` carried.
    """
    frame = pl.DataFrame({"g": np.repeat(np.arange(_GROUPS), _PER_GROUP)}).with_columns(
        **{p.column: (pl.lit(v) * _GROUP_SCALE).cast(p.dtype) for p, v in spec.pairs()}
    )
    agg = (
        frame.group_by("g", maintain_order=True)
        .agg(total=spec.build("name").sample(seed=_SEED).cast(pl.Float64).sum())
        .sort("g")
    )
    assert agg["total"].n_unique() > 1, f"{spec.name}: every group totalled the same with per-group parameters"


def test_over_partitions_keeps_the_full_length(spec: DistSpec) -> None:
    """`over` scatters each partition's draws back to row order, keeping the frame's length.

    The *second* partition is the discriminating one. Every group is seeded from local row indices
    `0..n-1`, so group 1 must repeat group 0's draws exactly; a sampler seeded from the global
    position would give it a different stream. Group 0 cannot show that, since its local and global
    indices are the same either way, and comparing the expression against a second evaluation of
    itself would only pin determinism.
    """
    frame = pl.DataFrame({"g": np.repeat(np.arange(_GROUPS), _PER_GROUP)})
    dist = spec.build("scalar")

    scattered = frame.select(z=dist.sample(seed=_SEED).over("g"))["z"]
    assert scattered.len() == _GROUPS * _PER_GROUP

    unpartitioned = frame.head(_PER_GROUP).select(z=dist.sample(seed=_SEED))["z"]
    second_group = scattered.slice(_PER_GROUP, _PER_GROUP)
    assert_series_equal(second_group, unpartitioned)


# --------------------------------------------------------------------------------------------------
# Bespoke sampler facts: a boundary the general contract cannot state, and the saturation regimes.
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
def test_a_cauchy_draw_tracks_loc_and_scale_through_its_quartiles(regime: Regime) -> None:
    """The quartiles sit at `loc -+ scale` and the median at `loc`, the parameter-order check the mean cannot make.

    The heavy tail disqualifies the mean: among these draws the farthest lies hundreds of scales out
    (asserted, as the witness that the tail is the Cauchy one), so the sample mean does not converge at
    any `n`. A sample quantile converges at the usual `1 / sqrt(n)`, with standard error
    `sqrt(q (1 - q)) / (pdf(x_q) sqrt(n))`: `pi scale / (2 sqrt n)` at the median and
    `sqrt(3) pi scale / (2 sqrt n)` at either quartile.
    """
    spec = SPECS_BY_NAME["cauchy"]
    loc, scale = spec.example
    drawn = _frame(_CLT_ROWS).select(z=spec.build(regime).sample(seed=_SEED))["z"]

    se_median = math.pi * scale / (2.0 * math.sqrt(_CLT_ROWS))
    se_quartile = math.sqrt(3.0) * se_median
    for q, want, se in ((0.25, loc - scale, se_quartile), (0.5, loc, se_median), (0.75, loc + scale, se_quartile)):
        got = drawn.quantile(q)
        assert isinstance(got, float)
        assert abs(got - want) <= _CLT_SIGMAS * se, f"quantile {q}: {got} is more than {_CLT_SIGMAS} SE from {want}"

    low, high = min_max(drawn)
    assert high - loc > 100.0 * scale
    assert loc - low > 100.0 * scale


def test_a_uniform_draw_stays_strictly_below_max_under_rounding() -> None:
    """Pins the half-open `[min, max)` clamp in `draw_half_open`, on both parameter routings.

    At this magnitude consecutive floats are `2.0` apart, so `min + (max - min) * u` rounds to
    exactly `max` for roughly half of all `u`; without the clamp this fails immediately.
    """
    mn, mx = 1e16, 1e16 + 2.0
    frame = _frame(2_000)
    fast = frame.select(u=Uniform(min=mn, max=mx).sample(seed=_SEED))
    per_row = frame.select(u=Uniform(min=pl.repeat(mn, pl.len()), max=pl.repeat(mx, pl.len())).sample(seed=_SEED))
    for got in (fast, per_row):
        assert got["u"].is_between(mn, mx, closed="left").all()


def test_a_one_point_discrete_uniform_always_draws_its_point() -> None:
    """`min == max` is a valid one-point mass, and the draw has nowhere else to go."""
    drawn = _frame(256).select(z=DiscreteUniform(min=4, max=4).sample(seed=_SEED))["z"]
    assert drawn.unique().to_list() == [4]


@pytest.mark.parametrize("p", [1e-3, 1e-6, 1e-9])
def test_a_geometric_draw_tracks_the_inverse_p_into_the_small_p_regime(p: float) -> None:
    """The regime where the trial count outgrows what a naive `floor(log(u) / log1p(-p))` resolves."""
    drawn = _frame(_CLT_ROWS).select(z=Geometric(p=p).sample(seed=_SEED))["z"]
    low, _high = min_max(drawn)
    assert low >= 1.0
    observed = drawn.cast(pl.Float64).mean()
    assert isinstance(observed, float)
    assert observed == pytest.approx(1.0 / p, rel=0.05)


def test_a_geometric_draw_saturates_where_the_trial_count_outgrows_uint64() -> None:
    """Below `p ~ 1 / u64::MAX` the true trial count is not representable, so the draw clamps.

    Clamping is the documented behaviour. What is pinned is that the clamp is *reached* rather than
    wrapping to a small count, and that it does not fire above the onset.
    """
    saturated = _frame(4_000).select(z=Geometric(p=1e-30).sample(seed=_SEED))["z"]
    assert saturated.unique().to_list() == [2**64 - 1]

    _low, high = min_max(_frame(4_000).select(z=Geometric(p=1e-3).sample(seed=_SEED))["z"])
    assert high < float(2**64 - 1)


def test_a_bernoulli_draw_at_an_extreme_p_is_deterministic() -> None:
    """`p = 0` never succeeds and `p = 1` always does, on the column routing as on the scalar one."""
    extreme = pl.DataFrame({"p": [0.0, 1.0, 0.0, 1.0]})
    got = extreme.select(z=Bernoulli(p=pl.col("p")).sample(seed=_SEED))["z"]
    assert got.to_list() == [False, True, False, True]


def test_a_multi_column_parameter_draws_one_column_each() -> None:
    """`pl.col("p1", "p2")` is one expression naming two columns; the sampler answers both.

    The output-name contract in `naming_test.py` is what makes this work, and this is where the
    result of it is observable.
    """
    frame = pl.DataFrame({"p1": [0.3, 0.6], "p2": [0.4, 0.9]})
    got = frame.with_columns(Bernoulli(p=pl.col("p1", "p2")).sample(seed=_SEED).name.suffix("_b"))
    assert got.schema == pl.Schema(frame.schema | {"p1_b": pl.Boolean(), "p2_b": pl.Boolean()})


def test_a_missing_column_name_is_reported_at_evaluation(spec: DistSpec) -> None:
    """Column resolution happens when the expression runs, not in `__init__`."""
    first = spec.parameters[0]
    kwargs: dict[str, object] = {p.name: p.cast(v) for p, v in spec.pairs()}
    kwargs[first.name] = "does_not_exist"
    dist = spec.cls(**kwargs)  # must not raise

    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        _frame(4).select(z=dist.sample(seed=_SEED))
