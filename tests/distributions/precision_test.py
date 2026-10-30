"""Numerical-regime facts that no shared contract and no scipy oracle reaches.

Each one is a single distribution in a single regime: a saturation zone where the linear form has
rounded away, an integer point past the `Float64` grid, or a corner where scipy is not a trustworthy
reference. They survive the per-distribution folders because they are genuinely per distribution, not
because a shared version has not been written yet.

Two neighbours: `tests/scipy_parity/<name>_test.py` owns the facts that *do* have a scipy oracle,
including its own "Regressions from `make audit`" sections, and
[`identities_test.py`](identities_test.py) owns the algebraic reductions between distributions.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import TYPE_CHECKING, TypeVar

import polars as pl
import pytest
from scipy.stats import binom as scipy_binom
from scipy.stats import cauchy as scipy_cauchy

from polars_stats import Beta, Binomial, Cauchy, DiscreteUniform, Exponential, Gamma, Geometric, Pareto, Weibull
from tests._registry import DRIVER_REGIMES, SPECS_BY_NAME

if TYPE_CHECKING:
    from polars_stats.distributions._base import _UnivariateDistribution
    from tests._registry import Regime

DistT = TypeVar("DistT", bound="_UnivariateDistribution")


def _spelled(cls: type[DistT], regime: Regime, *params: float) -> DistT:
    """`cls` with every parameter spelled in `regime`, so a test runs both Rust drivers."""
    dist = SPECS_BY_NAME[cls._distribution_name].build(regime, params=params)
    assert isinstance(dist, cls)
    return dist


# --------------------------------------------------------------------------------------------------
# Beta: the corner where a shape below 1 makes the density diverge and the sf approach 1.
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("shapes", "x", "expected"),
    [
        ((0.05, 0.05), 1e-16, 0.920450956743064),
        ((0.05, 0.05), 1e-20, 0.9498079469106636),
        ((0.05, 0.95), 1e-20, 0.9004107264756439),
    ],
    ids=["a=b=0.05, x=1e-16", "a=b=0.05, x=1e-20", "a=0.05 b=0.95, x=1e-20"],
)
def test_beta_sf_keeps_the_lower_corner_for_shapes_below_one(
    shapes: tuple[float, float], x: float, expected: float
) -> None:
    """`sf` near `0` for a shape below `1`, where the parity grid stops at `0.01`."""
    a, b = shapes
    result = pl.DataFrame({"x": [x]}).select(r=Beta(a=a, b=b).sf(pl.col("x")))["r"].item()
    assert result == pytest.approx(expected, rel=1e-14, abs=0.0)


@pytest.mark.parametrize("opposite", [0.5, 2.0, 100.0])
def test_beta_density_diverges_at_the_boundary_of_a_shape_below_one(opposite: float) -> None:
    """`pdf(0)` is `+inf` for `a < 1`, and `log_pdf` reports the same `inf` rather than a finite log."""
    df = pl.DataFrame({"x": [0.0]})
    got = df.select(pdf=Beta(a=0.5, b=opposite).pdf(pl.col("x")), log_pdf=Beta(a=0.5, b=opposite).log_pdf(pl.col("x")))
    assert got["pdf"].item() == math.inf
    assert got["log_pdf"].item() == math.inf


def test_beta_log_pdf_is_finite_in_the_upper_1e9_band() -> None:
    """The log-density is finite right up to the endpoint, so no value below 1 may return `-inf`.

    The expectation is the closed form rather than a recorded number, so this stays a live check.
    """
    x = 1.0 - 1e-9
    df = pl.DataFrame({"x": [x]})
    for a, b in [(2.0, 3.0), (0.05, 0.05), (5.0, 1e-3)]:
        ln_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        expected = (a - 1.0) * math.log(x) + (b - 1.0) * math.log(1.0 - x) - ln_beta
        result = df.select(r=Beta(a=a, b=b).log_pdf(pl.col("x")))["r"].item()
        assert result == pytest.approx(expected, rel=1e-13, abs=0.0)


def test_binomial_pmf_does_not_round_a_near_one_p_to_the_degenerate_case() -> None:
    """`p = 1 - 1e-10` at `n = 10**6`: `pmf(n)` is below `1`, not the `p == 1` point mass."""
    n, p = 10**6, 1 - 1e-10
    result = pl.DataFrame({"_": [0]}).select(v=Binomial(n, p).pmf(float(n))).item(0, "v")
    assert result < 1.0
    assert result == pytest.approx((1 - 1e-10) ** n, rel=1e-14)


# --------------------------------------------------------------------------------------------------
# Exponential and Geometric: log-scale saturation, where the linear form has already rounded to 1.
# --------------------------------------------------------------------------------------------------


def test_exponential_log_cdf_stable_in_deep_right_tail() -> None:
    """At `rate * x = 40`, `1 - exp(-rate * x)` rounds to exactly `1.0` and a naive `log(cdf)` is `0.0`.

    The `log1p(-sf)` form keeps the true value `log(1 - exp(-40)) ~= -exp(-40)`.
    """
    result = pl.DataFrame({"x": [40.0]}).select(r=Exponential(rate=1.0).log_cdf(pl.col("x")))["r"].item()
    assert result < 0.0
    assert result == pytest.approx(-math.exp(-40.0), rel=1e-6)


def test_geometric_log_cdf_tiny_p_keeps_full_precision() -> None:
    """The cdf sits a hair below 1 for a tiny `p`, so its linear log collapses to `0.0`.

    Entering through `log1p(-p)` keeps the whole regime: here `log_cdf` reads `log(k * p)`.
    """
    result = pl.DataFrame({"_": [0]}).select(v=Geometric(p=1e-17).log_cdf(1000)).item(0, "v")
    assert result == pytest.approx(math.log(1000 * 1e-17), rel=1e-12)


def test_geometric_log_pmf_tiny_p_keeps_full_precision() -> None:
    """`log(1 - p)` collapses to `0.0` for a tiny `p`; `log1p(-p)` keeps the whole small-`p` regime."""
    result = pl.DataFrame({"_": [0]}).select(v=Geometric(p=1e-17).log_pmf(1000)).item(0, "v")
    assert result == pytest.approx(math.log(1e-17))


@pytest.mark.parametrize("value", [100, 10_000, 1_000_000])
def test_geometric_log_cdf_deep_tail_does_not_underflow(value: int) -> None:
    """`cdf = 1 - (1-p)**k` rounds to exactly `1.0` long before here, so a naive log would answer `0`."""
    p = 0.001
    result = pl.DataFrame({"_": [0]}).select(v=Geometric(p=p).log_cdf(value)).item(0, "v")
    assert result == pytest.approx(math.log1p(-math.exp(value * math.log1p(-p))))


# --------------------------------------------------------------------------------------------------
# DiscreteUniform: an integer support past the `Float64` grid, and the `N * (1 / N) != 1` counts.
#
# `N = 49, 98, 103, 107` are the first four support counts where `N * (1 / N) != 1.0`, so the curated
# pairs below fail if `cdf(max)` is a clamped product rather than an explicit `1.0` branch.
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("lo", "hi"), [(1, 6), (-5, 9), (0, 100), (3, 3), (0, 48), (0, 97), (0, 102), (0, 106)])
def test_discreteuniform_cdf_max_is_exactly_one(lo: int, hi: int) -> None:
    """`cdf(max)` is exactly 1, the visible difference from scipy's exclusive upper bound."""
    got = pl.DataFrame({"x": [float(hi)]}).select(r=DiscreteUniform(min=lo, max=hi).cdf("x"))["r"][0]
    assert got == 1.0


def test_discreteuniform_cdf_max_is_exactly_one_for_every_support_count_up_to_4000() -> None:
    """The column-routed sweep behind the curated pairs, covering all 483 counts that lose the product."""
    counts = range(1, 4001)
    df = pl.DataFrame(
        {"lo": [0] * len(counts), "hi": [c - 1 for c in counts], "x": [float(c - 1) for c in counts]},
        schema_overrides={"lo": pl.Int64, "hi": pl.Int64},
    )
    result = df.select(r=DiscreteUniform(min=pl.col("lo"), max=pl.col("hi")).cdf("x"))["r"]
    assert result.to_list() == [1.0] * len(counts)


def test_discreteuniform_sf_below_min_is_exactly_one_for_every_support_count_up_to_4000() -> None:
    """The `cdf(max) == 1` mirror: below the support the survival mass is exactly 1, not `N * (1/N)`."""
    counts = range(1, 4001)
    df = pl.DataFrame(
        {"lo": [0] * len(counts), "hi": [c - 1 for c in counts], "x": [-1.0] * len(counts)},
        schema_overrides={"lo": pl.Int64, "hi": pl.Int64},
    )
    result = df.select(r=DiscreteUniform(min=pl.col("lo"), max=pl.col("hi")).sf("x"))["r"]
    assert result.to_list() == [1.0] * len(counts)


@pytest.mark.parametrize("method", ["cdf", "log_cdf"])
def test_discreteuniform_saturates_for_an_int_value_past_int64_half(method: str) -> None:
    """The `value >= max` branch discards a wrapped count, so the `int` and `float` spellings agree."""
    dist = DiscreteUniform(min=0, max=10)
    point = 2**63 - 1
    unit = pl.DataFrame({"_": [0]})
    from_int = unit.select(v=getattr(dist, method)(point)).item(0, "v")
    from_float = unit.select(v=getattr(dist, method)(float(point))).item(0, "v")
    assert from_int == (1.0 if method == "cdf" else 0.0)
    assert from_int == from_float


@pytest.mark.parametrize("offset", [0, 1, 5, 9, 10])
def test_discreteuniform_is_exact_for_an_int_value_in_a_narrow_support_past_the_float_grid(offset: int) -> None:
    """An `int` evaluation point resolves every support point where a `Float64` one cannot.

    `cdf` accumulates `floor(value) - min + 1` and `sf` subtracts `max - floor(value)`, both in
    `Int64`, so both stay exact where the point itself is not representable as a distinct `Float64`.
    """
    lo = 2**62
    hi = lo + 10
    n = hi - lo + 1
    dist = DiscreteUniform(min=lo, max=hi)
    unit = pl.DataFrame({"_": [0]})

    assert unit.select(v=dist.cdf(lo + offset)).item(0, "v") == pytest.approx(float(Fraction(offset + 1, n)), rel=1e-15)
    assert unit.select(v=dist.sf(lo + offset)).item(0, "v") == pytest.approx(
        float(Fraction(hi - lo - offset, n)), rel=1e-15
    )


@pytest.mark.parametrize("n", [10_001, 10**5, 10**7, 10**9])
@pytest.mark.parametrize("method", ["log_cdf", "log_sf"])
def test_discreteuniform_log_saturation_zone_reads_the_tail(method: str, n: int) -> None:
    """One step inside each end, the log reads `log1p(-1/N)` rather than the log of a ratio rounded to 1."""
    lo, hi = 0, n - 1
    point = float(hi - 1) if method == "log_cdf" else float(lo)
    got = pl.DataFrame({"x": [point]}).select(r=getattr(DiscreteUniform(min=lo, max=hi), method)("x"))["r"][0]
    assert got == pytest.approx(math.log1p(-1.0 / n), rel=1e-15)


# --------------------------------------------------------------------------------------------------
# The discrete inverses at a step boundary, where the tie-break is the platform's `exp` and `log1p`
# rather than the rule. Pinned rather than fixed; see docs/explanation/accuracy.md,
# "Discrete `ppf` / `isf` at a step boundary".
# --------------------------------------------------------------------------------------------------


def _smallest_geometric_point(p: float, quantile: float) -> int:
    """Smallest `k >= 1` with `cdf(k) >= quantile`, evaluated without rounding.

    `1 - (1 - p)**k >= q` over `Fraction`s, so this is the definition itself rather than a second
    floating-point implementation of it to compare against.
    """
    failure, target = 1 - Fraction(p), Fraction(quantile)
    k, tail = 1, failure
    while 1 - tail < target:
        k += 1
        tail *= failure
    return k


def test_geometric_ppf_matches_the_closed_form_for_a_small_p() -> None:
    """An interior quantile where a naive `log(1 - q) / log(1 - p)` loses digits on both sides."""
    p, q = 1e-13, 1e-11
    expected = math.ceil(math.log1p(-q) / math.log1p(-p))
    got = pl.DataFrame({"_": [0]}).select(v=Geometric(p=p).ppf(q)).item(0, "v")
    assert got == expected


@pytest.mark.parametrize(("p", "step"), [(0.1, 10), (0.3, 4), (0.5, 20), (0.05, 13)])
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_geometric_ppf_at_an_exact_step_boundary_can_miss_by_one(p: float, step: int, offset: int) -> None:
    """On a step the answer is the definition's support point or a neighbour, and no further.

    The tie-break compares `k * log1p(-p)` against `log1p(-q)` rather than re-deriving `cdf(k)`.
    That is deliberate (1997 boundary probes: 357 disagreements with exact arithmetic against 490 for
    the alternative), and the price is that `ppf` and `cdf` are not exact mutual inverses on a step:
    the two roundings decide the last bit there. Which side they fall on is the platform's `exp` and
    `log1p`, not the rule, so only the bound is portable and only the bound is pinned. At `p = 0.1`
    one ulp above `cdf(10)`, Apple's libm answers `10` and glibc `11`, because their `exp` puts
    `cdf(10)` itself an ulp apart.
    """
    unit = pl.DataFrame({"_": [0]})
    cdf_at_step = unit.select(v=Geometric(p=p).cdf(float(step))).item(0, "v")
    quantile = cdf_at_step if offset == 0 else math.nextafter(cdf_at_step, float(offset > 0))

    got = unit.select(v=Geometric(p=p).ppf(quantile)).item(0, "v")
    assert abs(got - _smallest_geometric_point(p, quantile)) <= 1


@pytest.mark.parametrize(("p", "step"), [(0.1, 10), (0.3, 4), (0.5, 20), (0.05, 13)])
def test_geometric_ppf_at_a_step_boundary_agrees_across_routings(p: float, step: int) -> None:
    """A column `p` takes the same tie-break as a Python-float `p`, on a frame longer than one row.

    A one-row frame is the one length at which a Polars expression on a literal `p` folds like a
    column one, so a scalar-only probe cannot see the two routings drift apart at a step.
    """
    unit = pl.DataFrame({"_": [0]})
    cdf_at_step = unit.select(v=Geometric(p=p).cdf(float(step))).item(0, "v")
    quantiles = [math.nextafter(cdf_at_step, 0.0), cdf_at_step, math.nextafter(cdf_at_step, 1.0)]
    frame = pl.DataFrame({"q": quantiles, "p": [p] * len(quantiles)})

    got = frame.select(scalar=Geometric(p=p).ppf(pl.col("q")), column=Geometric(p=pl.col("p")).ppf(pl.col("q")))
    assert got["scalar"].to_list() == got["column"].to_list()
    for quantile, k in zip(quantiles, got["scalar"].to_list(), strict=True):
        assert abs(k - _smallest_geometric_point(p, quantile)) <= 1


def test_geometric_isf_deep_tail_keeps_full_precision() -> None:
    """`ppf(1 - q)` would quantise the tail mass to the spacing of `1.0` before the inverse runs."""
    p, q = 0.5, 1e-300
    expected = math.ceil(math.log(q) / math.log1p(-p))
    got = pl.DataFrame({"_": [0]}).select(v=Geometric(p=p).isf(q)).item(0, "v")
    assert got == expected


def test_geometric_isf_inverts_sf() -> None:
    """`sf(isf(q)) <= q < sf(isf(q) - 1)`, the definition, at quantiles the parity grid never reaches."""
    unit = pl.DataFrame({"_": [0]})
    for p, q in [(0.1, 1e-4), (0.3, 1e-8), (0.7, 0.9)]:
        k = unit.select(v=Geometric(p=p).isf(q)).item(0, "v")
        assert unit.select(v=Geometric(p=p).sf(k)).item(0, "v") <= q
        assert unit.select(v=Geometric(p=p).sf(k - 1)).item(0, "v") > q


def test_geometric_isf_overshoots_by_one_at_an_exact_step_boundary() -> None:
    """At `q = sf(1)` the answer is `2`, where `sf(1) <= q` already makes `1` the definition's answer.

    The direction is pinnable here, unlike on the `ppf` twin. `sf(1)` sits so close to `1` that the
    `log` back out of it is quantised to the spacing of `1.0`, which is `1e-8` *relative* at this
    `p`: the ratio misses the integer by `5e-9` rather than by a last bit, so every libm agrees on
    which side it falls. One ulp above the boundary the answer is `1` as expected, so the miss is
    one-sided in `q`, not a shifted support. The column routing takes the same tie-break.
    """
    p, support_floor, next_point = 1e-8, 1.0, 2.0
    unit = pl.DataFrame({"_": [0]})
    sf_at_floor = unit.select(v=Geometric(p=p).sf(support_floor)).item(0, "v")
    quantiles = [sf_at_floor, math.nextafter(sf_at_floor, 0.0), math.nextafter(sf_at_floor, 1.0)]
    want = [next_point, next_point, support_floor]

    frame = pl.DataFrame({"q": quantiles, "p": [p] * len(quantiles)})
    got = frame.select(scalar=Geometric(p=p).isf(pl.col("q")), column=Geometric(p=pl.col("p")).isf(pl.col("q")))
    assert got["scalar"].to_list() == want
    assert got["column"].to_list() == want


# The grid below displaces every step edge by half an ulp, where the rational answer is unambiguous
# and a bare `floor` of the rounded product loses a support point. The exactly representable edges are
# excluded deliberately: there `q` is the nearest float to `k / N` rather than `k / N` itself, so the
# rational contract and inverting this library's own `sf` disagree.
@pytest.mark.parametrize(("lo", "hi"), [(1, 6), (0, 7), (-5, 9), (0, 100), (0, 999)])
def test_discreteuniform_isf_matches_the_exact_closed_form_off_the_representable_edges(lo: int, hi: int) -> None:
    """Exact agreement with rational `max - floor(q * N)` on both sides of every step."""
    n = hi - lo + 1
    edges = [k / n for k in range(1, n)]
    quantiles = sorted({e - 2**-53 for e in edges} | {e + 2**-53 for e in edges})
    got = pl.DataFrame({"q": quantiles}).select(r=DiscreteUniform(min=lo, max=hi).isf("q"))["r"]
    expected = [float(min(max(hi - math.floor(Fraction(q) * n), lo), hi)) for q in quantiles]
    assert got.to_list() == expected


@pytest.mark.parametrize(("lo", "hi"), [(1, 6), (0, 7), (-5, 9), (0, 100), (-20, -10), (3, 3), (0, 999)])
def test_discreteuniform_isf_round_trips_every_support_point_to_within_one_step(lo: int, hi: int) -> None:
    """`isf(sf(x))` is `x` or the point above it, where `_sf`'s rounded `1 / N` and the probe's division disagree."""
    d = DiscreteUniform(min=lo, max=hi)
    points = pl.DataFrame({"x": [float(p) for p in range(lo, hi + 1)]})
    round_trip = points.select(x=pl.col("x"), rt=d.isf(d.sf("x")))
    assert round_trip.select(((pl.col("rt") == pl.col("x")) | (pl.col("rt") == pl.col("x") + 1)).all()).item()
    # The survival contract too; alone it is vacuous, since `sf(max) == 0` satisfies it for every `q`.
    quantiles = points.select(q=d.sf("x"))
    assert quantiles.select((d.sf(d.isf("q")) <= pl.col("q")).all()).item()


# --------------------------------------------------------------------------------------------------
# The moments at the edges of their own arithmetic: an `Int64` midpoint that must not wrap, a mass
# term that underflows out of a support sum, and two deliberate divergences from scipy.
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("moment", ["mean", "median"])
@pytest.mark.parametrize(("lo", "hi"), [(2**62, 2**62 + 10), (2**62, 2**63 - 1), (-(2**61), 2**61 + 1)])
def test_discreteuniform_midpoint_does_not_wrap_near_the_int64_ceiling(moment: str, lo: int, hi: int) -> None:
    """`min + max` overflows `Int64` well inside the validated range; the oracle is the exact midpoint.

    `mean` and `median` compute it independently, so both are probed.
    """
    got = pl.DataFrame({"_": [0]}).select(v=getattr(DiscreteUniform(min=lo, max=hi), moment)()).item(0, "v")
    assert got == float(Fraction(lo + hi, 2))


@pytest.mark.parametrize(
    ("lo", "hi"),
    [(-4409513171799557951, 4409659924503471479), (-(2**61), 2**61 + 1), (-(2**62) + 3, 2**62 - 7)],
)
def test_discreteuniform_mean_is_exact_for_bounds_straddling_zero(lo: int, hi: int) -> None:
    """Summing the bounds in `Float64` rounds both before they cancel; the `Int64` midpoint does not."""
    exact = Fraction(lo + hi, 2)
    # Each midpoint is representable, so any inequality below is the cancellation defect, not rounding.
    assert Fraction(float(exact)) == exact
    got = pl.DataFrame({"_": [0]}).select(v=DiscreteUniform(min=lo, max=hi).mean()).item(0, "v")
    assert got == float(exact)


@pytest.mark.parametrize(("lo", "hi", "midpoint", "ppf_half"), [(1, 6, 3.5, 3.0), (1, 8, 4.5, 4.0)])
def test_discreteuniform_median_is_the_midpoint_not_ppf_half(
    lo: int, hi: int, midpoint: float, ppf_half: float
) -> None:
    """A documented divergence from scipy, which reports the support point `ppf(0.5)` answers."""
    dist = DiscreteUniform(min=lo, max=hi)
    got = pl.DataFrame({"_": [0]}).select(median=dist.median(), ppf=dist.ppf(0.5))
    assert got["median"][0] == midpoint
    assert got["ppf"][0] == ppf_half


def test_binomial_median_disagrees_with_floor_np_where_scipy_does() -> None:
    """Guard the convention choice: `floor(10 * 0.75)` is `7`, but the scipy / `ppf(0.5)` median is `8`."""
    scipy_median = 8.0
    got = pl.DataFrame({"_": [0]}).select(v=Binomial(10, 0.75).median()).item(0, "v")
    assert got == scipy_median


@pytest.mark.parametrize(("n", "p"), [(50, 1e-8), (1000, 0.999)])
def test_binomial_entropy_survives_an_underflowing_mass_term(n: int, p: float) -> None:
    """The Rust support sum drops a term whose `pmf` underflows, and still totals the right entropy.

    A parameter regime the parity `_NS` / `_PROBS` grid never reaches, so it is oracled here against
    scipy directly rather than through `scipy_parity/_harness.py`.
    """
    got = pl.DataFrame({"_": [0]}).select(v=Binomial(n, p).entropy()).item(0, "v")
    assert got == pytest.approx(float(scipy_binom.entropy(n, p)), rel=1e-8)


# --------------------------------------------------------------------------------------------------
# Cauchy: the tails where `0.5 - atan(z) / pi` cancels and `tan(pi (q - 1/2))` loses a small `q`.
#
# scipy spells `sf` as `0.5 - arctan(z) / pi` and `logsf` as its log: absolute error `~1e-17` on a
# value `~1 / (pi z)`, so `3e-10` relative at `z = 1e6` and worse beyond. `cauchy.rs` reads the tail
# through `atan2(1, z)`, and the oracle here is that identity in plain `math`, exact wherever `1 / z`
# is. The inverses are oracled against `cot(pi q) = 1 / (pi q) - pi q / 3 + ...`, whose dropped term
# is below `1e-16` relative for every quantile here.
# --------------------------------------------------------------------------------------------------

_CAUCHY_TAIL_PARAMS = [(0.0, 1.0), (1.5, 2.0), (100.0, 1e-3)]


@pytest.mark.parametrize(("loc", "scale"), _CAUCHY_TAIL_PARAMS, ids=str)
@pytest.mark.parametrize("z", [1e6, 1e12, 1e100, 1e300], ids=lambda z: f"z={z}")
def test_cauchy_tail_methods_keep_relative_precision_where_the_arctangent_cancels(
    loc: float, scale: float, z: float
) -> None:
    """`sf`, `log_sf` far right and `cdf`, `log_cdf` far left hold `1e-14` relative; the near-certain
    side of each log method reads `log1p` of the far tail rather than the log of a value rounded to `1`."""
    dist = Cauchy(loc=loc, scale=scale)
    frame = pl.DataFrame({"upper": [loc + z * scale], "lower": [loc - z * scale]})
    got = frame.select(
        sf=dist.sf("upper"),
        cdf=dist.cdf("lower"),
        log_sf=dist.log_sf("upper"),
        log_cdf=dist.log_cdf("lower"),
        log_cdf_near_one=dist.log_cdf("upper"),
        log_sf_near_one=dist.log_sf("lower"),
    )
    tail = math.atan(1.0 / z) / math.pi
    assert got["sf"].item() == pytest.approx(tail, rel=1e-14, abs=0.0)
    assert got["cdf"].item() == pytest.approx(tail, rel=1e-14, abs=0.0)
    assert got["log_sf"].item() == pytest.approx(math.log(tail), rel=1e-14, abs=0.0)
    assert got["log_cdf"].item() == pytest.approx(math.log(tail), rel=1e-14, abs=0.0)
    assert got["log_cdf_near_one"].item() == pytest.approx(math.log1p(-tail), rel=1e-14, abs=0.0)
    assert got["log_sf_near_one"].item() == pytest.approx(math.log1p(-tail), rel=1e-14, abs=0.0)


@pytest.mark.parametrize(("loc", "scale"), _CAUCHY_TAIL_PARAMS, ids=str)
@pytest.mark.parametrize("q", [1e-300, 1e-100, 1e-20, 1e-9], ids=lambda q: f"q={q}")
def test_cauchy_inverses_keep_relative_precision_in_the_tails(loc: float, scale: float, q: float) -> None:
    """`ppf(q)` is `loc - scale / (pi q)` and `isf(q)` its mirror, to `1e-14` relative, down to the smallest `q`.

    The textbook `loc + scale * tan(pi (q - 1/2))`, which is scipy's, forms `q - 1/2` first: that drops
    the low bits of a small `q`, and the tangent near `-pi / 2` magnifies what was dropped by
    `1 / (pi q)^2`, to `7e-8` relative at `q = 1e-9`.
    """
    dist = Cauchy(loc=loc, scale=scale)
    got = pl.DataFrame({"q": [q]}).select(ppf=dist.ppf("q"), isf=dist.isf("q"))
    tail_distance = scale / (math.pi * q)
    assert got["ppf"].item() == pytest.approx(loc - tail_distance, rel=1e-14, abs=0.0)
    assert got["isf"].item() == pytest.approx(loc + tail_distance, rel=1e-14, abs=0.0)


# `scale` far enough out that an intermediate leaves `float64` range while the density does not. Each
# id names the intermediate: the oracles divide stepwise so the reference never forms it either.
_CAUCHY_EXTREME_SCALES = [
    # `pi * scale` overflows above `scale ~ 5.7e307`, taking the peak to `0` and `ln(pi scale)` to `inf`.
    pytest.param(1e308, 0.0, 1.0 / math.pi / 1e308, -(math.log(math.pi) + math.log(1e308)), id="pi-times-scale"),
    # The peak `1 / (pi scale)` is unrepresentable below `scale ~ 1.77e-309`, and the standardised
    # point has overflowed too, so the density comes off the raw distance.
    pytest.param(1e-310, 1.0, 1e-310 / math.pi, math.log(1e-310) - math.log(math.pi), id="the-peak-and-z"),
    # The peak alone: `z` is finite here, and the true density is an ordinary number.
    pytest.param(1e-310, 1e-156, 1.0 / math.pi / (1.0 + 1e308) / 1e-310, None, id="the-peak-alone"),
]


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
@pytest.mark.parametrize(("scale", "point", "pdf", "log_pdf"), _CAUCHY_EXTREME_SCALES)
def test_cauchy_density_survives_an_intermediate_leaving_float64_range(
    regime: Regime, scale: float, point: float, pdf: float, log_pdf: float | None
) -> None:
    """Only a density outside `float64` range is lost; no intermediate takes one with it.

    Each row overflows a different intermediate while the answer stays representable. The third is
    the widest: with the peak overflowed and `z` finite, a single `peak / (1 + z^2)` reads `inf` for
    every point inside `|z| ~ 1.3e154`, however small its density.
    """
    dist = _spelled(Cauchy, regime, 0.0, scale)
    got = pl.DataFrame({"x": [point] * 4}).select(pdf=dist.pdf("x"), log_pdf=dist.log_pdf("x"))
    assert got["pdf"][0] == pytest.approx(pdf, rel=1e-13, abs=0.0)
    if log_pdf is not None:
        assert got["log_pdf"][0] == pytest.approx(log_pdf, rel=1e-14)


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
@pytest.mark.parametrize("distance", [1e24, 1e100, 1e300], ids=lambda d: f"d={d}")
def test_cauchy_log_tails_survive_the_tail_mass_itself_underflowing(regime: Regime, distance: float) -> None:
    """`scale / |d|` below the smallest subnormal makes the tail mass `0`, but its log is ordinary.

    `Cauchy(0, 1e-300).log_cdf(-1e24)` is `-747.18`, not `-inf`: below the smallest normal the tail is
    `scale / (pi |d|)` to far beyond `float64`, so the log is read off the three factors instead.
    `cdf` genuinely has no answer in this regime and stays `0.0`.
    """
    scale = 1e-300
    dist = _spelled(Cauchy, regime, 0.0, scale)
    frame = pl.DataFrame({"lower": [-distance] * 4, "upper": [distance] * 4})
    got = frame.select(log_cdf=dist.log_cdf("lower"), log_sf=dist.log_sf("upper"), cdf=dist.cdf("lower"))
    want = math.log(scale) - math.log(distance) - math.log(math.pi)
    assert got["log_cdf"][0] == pytest.approx(want, rel=1e-14)
    assert got["log_sf"][0] == pytest.approx(want, rel=1e-14)
    assert got["cdf"][0] == 0.0


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
def test_cauchy_tails_survive_a_scale_that_cannot_standardise_the_point(regime: Regime) -> None:
    """A `scale` small enough that `(value - loc) / scale` overflows still answers on every method.

    `Cauchy(0, 1e-300)` reaches that at `|value| > 1.8e8`, far inside the range where the true tail
    (`3.18e-311` here) is an ordinary subnormal. Standardising first would lose all four tail methods
    to a saturated `0.0` / `1.0` / `-inf`, and would make `pdf` `inf / inf`, a `NaN` the contract
    reserves for a `NaN` evaluation point.
    """
    scale = 1e-300
    dist = _spelled(Cauchy, regime, 0.0, scale)
    # Four rows: on a height-1 frame the `column` parameter is length 1 and `params_are_constant`
    # routes it back to the scalar path.
    frame = pl.DataFrame({"upper": [1e10] * 4, "lower": [-1e10] * 4})
    got = frame.select(
        sf=dist.sf("upper"),
        cdf=dist.cdf("lower"),
        log_sf=dist.log_sf("upper"),
        log_cdf=dist.log_cdf("lower"),
        pdf=dist.pdf("upper"),
        log_pdf=dist.log_pdf("upper"),
    )
    # `tail` is subnormal (3.18e-311, ~6.4e12 ulps above zero), so `rel=1e-14` would be six times
    # below one ulp of a libm `atan2` that only owes one; an ulp of the tail moves its log by 2e-16.
    tail = scale / (math.pi * 1e10)
    assert got["sf"][0] == pytest.approx(tail, rel=1e-12, abs=0.0)
    assert got["cdf"][0] == pytest.approx(tail, rel=1e-12, abs=0.0)
    assert got["log_sf"][0] == pytest.approx(math.log(tail), rel=1e-14, abs=0.0)
    assert got["log_cdf"][0] == pytest.approx(math.log(tail), rel=1e-14, abs=0.0)
    assert got["pdf"][0] == pytest.approx(scale / (math.pi * 1e20), rel=1e-13, abs=0.0)
    assert got["log_pdf"][0] == pytest.approx(math.log(scale) - math.log(math.pi) - 2.0 * math.log(1e10), rel=1e-14)


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
def test_cauchy_entropy_survives_a_scale_whose_four_pi_multiple_overflows(regime: Regime) -> None:
    """`log(4 pi scale)` is `inf` above `scale ~ 1.43e307`; summed as `log(4 pi) + log(scale)` it is not.

    The true entropy never exceeds `712.3` anywhere in `float64` range, so the overflow was purely in
    the intermediate. This is also how scipy spells it, which is why parity holds at both ends.
    """
    scale = 1e308
    dist = _spelled(Cauchy, regime, 0.0, scale)
    got = pl.DataFrame({"_": range(4)}).select(entropy=dist.entropy())["entropy"][0]
    assert got == pytest.approx(float(scipy_cauchy(loc=0.0, scale=scale).entropy()), rel=1e-14)


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
def test_cauchy_inverses_keep_a_negative_zero_quantile_on_the_negative_side(regime: Regime) -> None:
    """`-0.0` is inside `[0, 1]` for the range gate, so it reaches the quantile and must still mean `q = 0`.

    Without the `abs()` in `standard_quantile`, `PI * -0.0` is `-0.0`, its tangent is `-0.0`, and
    `-1.0 / -0.0` is `+inf`: `ppf(-0.0)` would return the upper support bound instead of the lower.
    A user column reaches `-0.0` through any `0.0 * -1` or a negated aggregate.
    """
    dist = _spelled(Cauchy, regime, 1.5, 2.0)
    got = pl.DataFrame({"q": [-0.0, 0.0]}).select(ppf=dist.ppf("q"), isf=dist.isf("q"))
    assert got["ppf"].to_list() == [-math.inf, -math.inf]
    assert got["isf"].to_list() == [math.inf, math.inf]


# --------------------------------------------------------------------------------------------------
# Pareto: a point just above `scale`, where `ln(x / scale)` has few digits left, and the regimes past
# where the survival mass, a single `exp`, the literal power, the excess `(x - scale) / scale` or the
# leading `shape / x` leaves `float64`. The oracles spell `pareto.rs`'s `ln_1p((x - scale) / scale)`
# through `math`.
# --------------------------------------------------------------------------------------------------

_PARETO_NEAR_SCALE = [(3.0, 0.5), (7.5, 3.0), (1e5, 50.0)]


@pytest.mark.parametrize(("scale", "shape"), _PARETO_NEAR_SCALE, ids=str)
@pytest.mark.parametrize("eps", [1e-8, 1e-12, 1e-14], ids=lambda e: f"eps={e}")
def test_pareto_cumulative_methods_keep_relative_precision_just_above_the_scale(
    scale: float, shape: float, eps: float
) -> None:
    """`cdf(scale (1 + eps))` is `shape eps` to first order and holds `1e-14` relative down to `eps = 1e-14`.

    The literal `1 - (scale / x) ** shape` rounds the ratio to `1 - eps` plus a few ulps of `1` before
    the power: `1.3e-8` relative at `eps = 1e-8` and `1.5e-2` at `1e-14` against a 50-digit reference.
    """
    dist = Pareto(scale=scale, shape=shape)
    x = scale * (1.0 + eps)
    t = math.log1p((x - scale) / scale)
    got = pl.DataFrame({"x": [x]}).select(
        cdf=dist.cdf("x"), log_cdf=dist.log_cdf("x"), sf=dist.sf("x"), log_sf=dist.log_sf("x")
    )
    assert got["cdf"].item() == pytest.approx(-math.expm1(-shape * t), rel=1e-14, abs=0.0)
    assert got["log_cdf"].item() == pytest.approx(math.log(-math.expm1(-shape * t)), rel=1e-14, abs=0.0)
    assert got["sf"].item() == pytest.approx(math.exp(-shape * t), rel=1e-14, abs=0.0)
    assert got["log_sf"].item() == pytest.approx(-shape * t, rel=1e-14, abs=0.0)


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
def test_pareto_log_tails_survive_the_survival_mass_underflowing(regime: Regime) -> None:
    """`Pareto(1, 100).log_sf(1e4)` is `-921.03` where `sf` is `1e-400`, below the smallest subnormal.

    `log_sf` is `-shape ln(x / scale)` exactly, with no linear value to underflow. `log_cdf` on the
    same row is `ln_1p(-sf)`, which rounds to `0` because the true `-1e-400` does.
    """
    dist = _spelled(Pareto, regime, 1.0, 100.0)
    frame = pl.DataFrame({"x": [1e4] * 4})
    got = frame.select(sf=dist.sf("x"), log_sf=dist.log_sf("x"), cdf=dist.cdf("x"), log_cdf=dist.log_cdf("x"))
    assert got["sf"][0] == 0.0
    assert got["log_sf"][0] == pytest.approx(-100.0 * math.log(1e4), rel=1e-15)
    assert got["cdf"][0] == 1.0
    assert got["log_cdf"][0] == 0.0


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
def test_pareto_inverses_survive_an_exponent_a_single_exp_cannot_hold(regime: Regime) -> None:
    """`Pareto(1e-300, 1).isf(1e-310)` is `1e10`: `exp(713.8)` alone is `inf`, `scale exp(t / 2) exp(t / 2)` is not.

    Both inverses are `scale exp(t)` with `t` the exponential quantile, which passes `709.8` long
    before the answer under a small `scale` leaves `float64`.
    """
    dist = _spelled(Pareto, regime, 1e-300, 1.0)
    got = pl.DataFrame({"q": [1e-310] * 4}).select(isf=dist.isf("q"))
    assert got["isf"][0] == pytest.approx(1e10, rel=1e-12, abs=0.0)

    dist = _spelled(Pareto, regime, 1e-300, 0.05)
    q = 1.0 - 1e-16
    t = -math.log1p(-q) / 0.05
    got = pl.DataFrame({"q": [q] * 4}).select(ppf=dist.ppf("q"))
    assert got["ppf"][0] == pytest.approx(math.exp(t + math.log(1e-300)), rel=1e-12, abs=0.0)


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
def test_pareto_density_survives_a_subnormal_survival_factor(regime: Regime) -> None:
    """`Pareto(1e-16, 100).pdf(x)` where `(scale / x) ** shape = e^-740` is an ordinary `2.6e-307`.

    A single `exp(-740)` is a subnormal with two digits left, and `shape / x` cannot restore what it
    dropped; the two half exponents never leave the normal range. The literal power answers `NaN`:
    `scale ** shape` and `x ** (shape + 1)` are both `0`.
    """
    scale, shape = 1e-16, 100.0
    x = scale * math.exp(7.4)
    dist = _spelled(Pareto, regime, scale, shape)
    got = pl.DataFrame({"x": [x] * 4}).select(pdf=dist.pdf("x"), log_pdf=dist.log_pdf("x"))
    log_pdf = math.log(shape) - math.log(x) - shape * math.log1p((x - scale) / scale)
    assert got["log_pdf"][0] == pytest.approx(log_pdf, rel=1e-14)
    assert got["pdf"][0] == pytest.approx(math.exp(log_pdf), rel=1e-12, abs=0.0)


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
def test_pareto_median_survives_an_exponent_a_single_exp_cannot_hold(regime: Regime) -> None:
    """`Pareto(1e-300, 7e-4).median()` is `1.1e130`, where `scale * exp(ln(2) / shape)` alone is `+inf`.

    `median` is the base class' `ppf(0.5)`, so it inherits that inverse's split exponent rather than
    spelling the closed form a second time in Polars arithmetic.
    """
    scale, shape = 1e-300, 7e-4
    got = pl.DataFrame({"_": range(4)}).select(median=_spelled(Pareto, regime, scale, shape).median())
    want = math.exp(math.log(2.0) / shape + math.log(scale))
    assert got["median"][0] == pytest.approx(want, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
def test_pareto_density_survives_a_shape_over_scale_ratio_past_float64(regime: Regime) -> None:
    """`Pareto(1e-300, 1e10).pdf(scale (1 + 1e-9))` is `4.5e305`, where the leading `shape / x` is `+inf`.

    The survival factor is applied to `shape` before the division, and it never exceeds `1` on the
    support, so the intermediate stays inside `float64` wherever the density itself does.
    """
    scale, shape = 1e-300, 1e10
    x = scale * (1.0 + 1e-9)
    dist = _spelled(Pareto, regime, scale, shape)
    got = pl.DataFrame({"x": [x] * 4}).select(pdf=dist.pdf("x"), log_pdf=dist.log_pdf("x"))
    log_pdf = math.log(shape) - math.log(x) - shape * math.log1p((x - scale) / scale)
    assert got["log_pdf"][0] == pytest.approx(log_pdf, rel=1e-14)
    assert got["pdf"][0] == pytest.approx(math.exp(log_pdf), rel=1e-12, abs=0.0)


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
def test_pareto_log_ratio_falls_back_to_a_difference_of_logs_when_the_excess_overflows(regime: Regime) -> None:
    """`Pareto(1e-300, 1).log_sf(1e10)` is `-713.8`: `(x - scale) / scale` is `+inf`, `ln(x) - ln(scale)` is not.

    Without the fallback `ln_1p(+inf)` would carry `+inf` into every method, so `log_sf` would be
    `-inf` and the density `0` where both are ordinary.
    """
    scale, shape = 1e-300, 1.0
    t = math.log(1e10) - math.log(scale)
    dist = _spelled(Pareto, regime, scale, shape)
    got = pl.DataFrame({"x": [1e10] * 4}).select(
        log_sf=dist.log_sf("x"), log_pdf=dist.log_pdf("x"), sf=dist.sf("x"), cdf=dist.cdf("x")
    )
    assert got["log_sf"][0] == pytest.approx(-shape * t, rel=1e-15, abs=0.0)
    assert got["log_pdf"][0] == pytest.approx(math.log(shape) - math.log(1e10) - shape * t, rel=1e-15, abs=0.0)
    assert got["sf"][0] > 0.0
    assert got["cdf"][0] == 1.0


# --------------------------------------------------------------------------------------------------
# Weibull: a point beside `scale` at a large shape, where the power magnifies the ratio's rounding;
# the regimes where the survival mass, a single `exp`, or the power leaves `float64`; the density at
# the origin; and the variance at a large shape, where the two gammas cancel.
# --------------------------------------------------------------------------------------------------

_WEIBULL_LARGE_SHAPE = [(1e4, 3.0), (1e6, 0.7), (1e8, 3.0)]


@pytest.mark.parametrize(("shape", "scale"), _WEIBULL_LARGE_SHAPE, ids=str)
@pytest.mark.parametrize("eps", [1e-9, 1e-12, 1e-14], ids=lambda e: f"eps={e}")
def test_weibull_cumulative_methods_keep_relative_precision_beside_the_scale(
    shape: float, scale: float, eps: float
) -> None:
    """`sf(scale (1 + eps))` is `exp(-exp(shape log1p(eps)))` and holds `1e-13` relative at `shape = 1e8`.

    The literal `(x / scale) ** shape` rounds the ratio before the power, which the exponent
    magnifies into `shape * 1.1e-16` relative: `1.1e-8` at `shape = 1e8`.
    """
    dist = Weibull(shape=shape, scale=scale)
    x = scale * (1.0 + eps)
    t = math.exp(shape * math.log1p((x - scale) / scale))
    got = pl.DataFrame({"x": [x]}).select(
        cdf=dist.cdf("x"), log_cdf=dist.log_cdf("x"), sf=dist.sf("x"), log_sf=dist.log_sf("x")
    )
    assert got["cdf"].item() == pytest.approx(-math.expm1(-t), rel=1e-13, abs=0.0)
    assert got["log_cdf"].item() == pytest.approx(math.log(-math.expm1(-t)), rel=1e-13, abs=0.0)
    assert got["sf"].item() == pytest.approx(math.exp(-t), rel=1e-13, abs=0.0)
    assert got["log_sf"].item() == pytest.approx(-t, rel=1e-13, abs=0.0)


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
def test_weibull_log_tails_survive_the_survival_mass_underflowing(regime: Regime) -> None:
    """`Weibull(2, 1).log_sf(30)` is `-900` where `sf` is `e ** -900`, below the smallest subnormal.

    `log_sf` is the power itself, with no linear value to underflow. `log_cdf` on the same row is
    `ln_1p(-sf)`, which rounds to `0` because the true `-e ** -900` does.
    """
    dist = _spelled(Weibull, regime, 2.0, 1.0)
    frame = pl.DataFrame({"x": [30.0] * 4})
    got = frame.select(sf=dist.sf("x"), log_sf=dist.log_sf("x"), cdf=dist.cdf("x"), log_cdf=dist.log_cdf("x"))
    assert got["sf"][0] == 0.0
    assert got["log_sf"][0] == pytest.approx(-900.0, rel=1e-15)
    assert got["cdf"][0] == 1.0
    assert got["log_cdf"][0] == 0.0


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
def test_weibull_log_cdf_survives_the_power_underflowing(regime: Regime) -> None:
    """`Weibull(100, 1e-5).log_cdf(1e-9)` is `100 ln(1e-4) = -921.03` where the power, and so `cdf`, is `1e-400`.

    Below `t = 2^-53` the log of the cdf is the log of the power to the last bit, and the power is
    never formed; `ln(-expm1(-t))` would be `ln(0) = -inf` from `t ~ 1e-308` down.
    """
    dist = _spelled(Weibull, regime, 100.0, 1e-5)
    got = pl.DataFrame({"x": [1e-9] * 4}).select(cdf=dist.cdf("x"), log_cdf=dist.log_cdf("x"), sf=dist.sf("x"))
    assert got["cdf"][0] == 0.0
    assert got["log_cdf"][0] == pytest.approx(100.0 * math.log(1e-4), rel=1e-15)
    assert got["sf"][0] == 1.0


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
def test_weibull_inverses_survive_an_exponent_a_single_exp_cannot_hold(regime: Regime) -> None:
    """`Weibull(0.004, 1e-300).isf(1e-10)` is `3.6e40`, where `(-ln q) ** 250` alone is `inf`.

    Both inverses are `scale exp(ln(t) / shape)` with `t` the unit exponential quantile, and at a
    small `shape` that exponent passes `709.8` long before the answer under a small `scale` leaves
    `float64`. Oracle: `mpmath` at 50 digits.
    """
    dist = _spelled(Weibull, regime, 0.004, 1e-300)
    got = pl.DataFrame({"q": [1e-10] * 4}).select(isf=dist.isf("q"), ppf=dist.ppf(1.0 - pl.col("q")))
    assert got["isf"][0] == pytest.approx(3.5803227233069835e40, rel=1e-12, abs=0.0)
    assert math.isfinite(got["ppf"][0])


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
def test_weibull_density_survives_a_subnormal_power(regime: Regime) -> None:
    """`Weibull(2, 1).pdf(1e-160)` is `2e-160`, where the power `x ** 2 = 1e-320` is a subnormal with four digits left.

    With the power's log folded into the exponent the subnormal `t` only enters as `exp(-t) = 1`;
    formed as `shape t exp(-t) / x` it would carry the subnormal's missing digits into an ordinary
    answer. The exponent route costs `|(shape - 1) ln(x / scale)| * 1.1e-16` relative, `4e-14` here.
    """
    dist = _spelled(Weibull, regime, 2.0, 1.0)
    got = pl.DataFrame({"x": [1e-160] * 4}).select(pdf=dist.pdf("x"), log_pdf=dist.log_pdf("x"))
    assert got["pdf"][0] == pytest.approx(2e-160, rel=1e-13, abs=0.0)
    assert got["log_pdf"][0] == pytest.approx(math.log(2.0) + math.log(1e-160), rel=1e-15)


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
@pytest.mark.parametrize(
    ("shape", "pdf", "log_pdf"),
    [(0.5, math.inf, math.inf), (1.0, 0.25, math.log(0.25)), (1.5, 0.0, -math.inf)],
    ids=["shape<1", "shape=1", "shape>1"],
)
def test_weibull_density_at_the_origin_follows_the_shape(
    regime: Regime, shape: float, pdf: float, log_pdf: float
) -> None:
    """`pdf(0)` is infinite below `shape = 1`, `1 / scale` at it and `0` above it, as `scipy.stats.weibull_min`.

    The origin is the one point where `(shape - 1) ln(x / scale)` is `0 * -inf`, so it is answered
    before the formula runs.
    """
    dist = _spelled(Weibull, regime, shape, 4.0)
    got = pl.DataFrame({"x": [0.0] * 4}).select(pdf=dist.pdf("x"), log_pdf=dist.log_pdf("x"))
    assert got["pdf"][0] == pdf
    assert got["log_pdf"][0] == pytest.approx(log_pdf, rel=1e-15)


# `mpmath` at 50 digits; the literal difference of gammas in `float64` is `1e-7` relative off at
# `shape = 1e4`, `12%` at `1e7` and `3.7x` at `1e8`.
_WEIBULL_UNIT_VARIANCE = [
    (100.0, 0.00016030491620026111),
    (1e4, 1.6445038762822376e-08),
    (1e6, 1.6449297637827162e-12),
    (1e8, 1.6449340238174553e-16),
]


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
@pytest.mark.parametrize(("shape", "unit_variance"), _WEIBULL_UNIT_VARIANCE, ids=lambda v: f"{v:.0e}")
def test_weibull_variance_keeps_its_digits_at_a_large_shape(regime: Regime, shape: float, unit_variance: float) -> None:
    """`Weibull(shape, 3).variance()` holds `1e-13` relative up to `shape = 1e8`.

    Both gammas tend to `1` and their difference to `pi ** 2 / (6 shape ** 2)`, so the literal
    subtraction loses `2 log10(shape)` digits; the log-gamma ratio is a series in `1 / shape` there.
    """
    dist = _spelled(Weibull, regime, shape, 3.0)
    got = pl.DataFrame({"_": range(4)}).select(variance=dist.variance(), std=dist.std())
    assert got["variance"][0] == pytest.approx(9.0 * unit_variance, rel=1e-13, abs=0.0)
    assert got["std"][0] == pytest.approx(3.0 * math.sqrt(unit_variance), rel=1e-13, abs=0.0)


# `mpmath` at 50 digits. `Gamma(1 + 1 / shape)` leaves `float64` at `shape ~ 5.9e-3` and
# `exp(ln Gamma(1 + 2 / shape) / 2)` at `shape ~ 3.3e-3`, both far above the `shape` where the moment
# itself does; a small enough `scale` is what keeps the product ordinary.
_WEIBULL_SMALL_SHAPE = [
    (0.004, 1e-300, 3.2328562609090149e192, 1.1045980381980728e267),
    (0.006, 1e-200, 2.7286180109907209e99, 8.4663490529360535e148),
    (0.01, 1e-40, 9.332621544394325e117, 2.8083053027845334e147),
]


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
@pytest.mark.parametrize(("shape", "scale", "mean", "std"), _WEIBULL_SMALL_SHAPE, ids=lambda v: f"{v:.0e}")
def test_weibull_gamma_moments_survive_a_shape_a_single_gamma_cannot_hold(
    regime: Regime, shape: float, scale: float, mean: float, std: float
) -> None:
    """`Weibull(0.004, 1e-300).mean()` is `3.23e192`, where `Gamma(251)` alone is `inf`.

    Both moments are `scale exp(t)` with `t` a log-gamma, so they saturate where the answer does
    rather than where the gamma function does, as `ppf` and `isf` do on the same parameterisation.
    At `1e-12` rather than the `1e-13` of the ordinary range: `exp` carries `|t| * 1.1e-16` relative,
    and `t` reaches `364` here, which is the price of the range.
    """
    dist = _spelled(Weibull, regime, shape, scale)
    got = pl.DataFrame({"_": range(4)}).select(mean=dist.mean(), std=dist.std())
    assert got["mean"][0] == pytest.approx(mean, rel=1e-12, abs=0.0)
    assert got["std"][0] == pytest.approx(std, rel=1e-12, abs=0.0)


# `mpmath` at 50 digits, straddling the `a = 1 / shape < 0.125` crossover at `shape = 8` where
# `variance` swaps the difference of log-gammas for its own series.
_WEIBULL_ACROSS_SERIES_CROSSOVER = [
    (7.9, 0.17966693964636126),
    (8.0, 0.17570847901744918),
    (8.1, 0.17188098654015858),
]


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
@pytest.mark.parametrize(("shape", "variance"), _WEIBULL_ACROSS_SERIES_CROSSOVER, ids=lambda v: f"{v:.2f}")
def test_weibull_variance_agrees_across_the_series_crossover(regime: Regime, shape: float, variance: float) -> None:
    """Both branches of the log-gamma ratio hold `1e-13` at `shape = 8`, so the seam has no step.

    `shape = 8` and below take the difference of log-gammas, which the cancellation leaves `5.3e-14`
    off here; `8.1` and above take the series, at `1.5e-15`. The crossover sits where the direct form
    still carries the digits the series keeps, and moving it down would widen that gap.
    """
    dist = _spelled(Weibull, regime, shape, 3.0)
    got = pl.DataFrame({"_": range(4)}).select(variance=dist.variance(), std=dist.std())
    assert got["variance"][0] == pytest.approx(variance, rel=1e-13, abs=0.0)
    assert got["std"][0] == pytest.approx(math.sqrt(variance), rel=1e-13, abs=0.0)


# --------------------------------------------------------------------------------------------------
# Gamma: the regularized incomplete gamma at the shapes where its textbook prefactor cancels (large)
# or its complement does (small), in the tails past `float64` range, and the entropy at a large shape.
# Every expected value is `mpmath` at 50 digits.
# --------------------------------------------------------------------------------------------------

# `(shape, x, cdf, sf)` five standard deviations either side of the mean. The textbook prefactor
# `shape ln t - t - ln Gamma(shape)` rounds terms of order `shape ln shape` that cancel to `ln(shape) / 2`,
# `1e-7` relative at `shape = 1e8`.
_GAMMA_LARGE_SHAPE = [
    (1e6, 995000.0, 2.7495803592700708e-7, 0.99999972504196407),
    (1e6, 1005000.0, 0.99999970125098599, 2.9874901401146349e-7),
    (1e8, 99950000.0, 2.8546421399586261e-7, 0.999999714535786),
    (1e8, 100050000.0, 0.99999971215703131, 2.8784296868527811e-7),
]


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
@pytest.mark.parametrize(("shape", "x", "cdf", "sf"), _GAMMA_LARGE_SHAPE, ids=lambda v: f"{v:.6g}")
def test_gamma_cumulative_methods_keep_their_digits_at_a_large_shape(
    regime: Regime, shape: float, x: float, cdf: float, sf: float
) -> None:
    """Five standard deviations from the mean, `cdf` and `sf` hold `1e-13` relative up to `shape = 1e8`.

    From `shape = 10` the prefactor is `shape (ln(t / shape) - t / shape + 1)` less Stirling's remainder,
    with the bracket read off `(t - shape) / shape`, so no term of order `shape ln shape` is rounded.
    """
    dist = _spelled(Gamma, regime, shape, 1.0)
    got = pl.DataFrame({"x": [x] * 4}).select(cdf=dist.cdf("x"), sf=dist.sf("x"), log_cdf=dist.log_cdf("x"))
    assert got["cdf"][0] == pytest.approx(cdf, rel=1e-13, abs=0.0)
    assert got["sf"][0] == pytest.approx(sf, rel=1e-13, abs=0.0)
    assert got["log_cdf"][0] == pytest.approx(math.log1p(-sf) if sf < cdf else math.log(cdf), rel=1e-13, abs=0.0)


# `(shape, x, sf, log_sf)`: `1 - P` would carry `P`'s `1.1e-16` absolute into an upper tail of `1e-8` and below.
_GAMMA_SMALL_SHAPE = [
    (1e-8, 0.5, 5.5977359770995871e-9, -19.000903610756109),
    (1e-6, 1e-3, 6.33152014144515e-6, -11.969970201911767),
    (1e-8, 3.0, 1.3048381341654404e-10, -22.75977193198106),
]


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
@pytest.mark.parametrize(("shape", "x", "sf", "log_sf"), _GAMMA_SMALL_SHAPE, ids=lambda v: f"{v:.3g}")
def test_gamma_upper_tail_keeps_its_digits_at_a_small_shape(
    regime: Regime, shape: float, x: float, sf: float, log_sf: float
) -> None:
    """At a small shape nearly all the mass sits at `0`, so `sf` is the small tail at every ordinary point.

    Below `x = 1.1` it is DLMF 8.7.3, whose leading `1 - x ** shape / Gamma(1 + shape)` is `-expm1` of a
    log formed from `ln Gamma(1 + shape)`'s own Taylor series: statrs' `ln_gamma(1.0 + shape)` rounds
    `1 + shape` first, which alone is `5e-8` relative on the tail at `shape = 1e-8`.
    """
    dist = _spelled(Gamma, regime, shape, 1.0)
    got = pl.DataFrame({"x": [x] * 4}).select(sf=dist.sf("x"), log_sf=dist.log_sf("x"))
    assert got["sf"][0] == pytest.approx(sf, rel=1e-14, abs=0.0)
    assert got["log_sf"][0] == pytest.approx(log_sf, rel=1e-14, abs=0.0)


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
@pytest.mark.parametrize(
    ("shape", "rate", "x", "log_sf"),
    [(2.5, 1.0, 1e3, -989.92155032737345), (0.3, 4.0, 5e2, -2006.4167794805587), (1e4, 1.0, 2e4, -3074.0525113731376)],
    ids=["shape=2.5", "shape=0.3", "shape=1e4"],
)
def test_gamma_log_sf_survives_the_survival_mass_underflowing(
    regime: Regime, shape: float, rate: float, x: float, log_sf: float
) -> None:
    """`log_sf` is the log prefactor plus the log of the continued fraction, finite where `sf` is `e ** -3074`.

    `cdf` on the same row is exactly `1`: its complement's `-expm1` would be `0 * inf` past `|ln sf| ~ 1420`,
    so a small enough tail is subtracted from `1` directly.
    """
    dist = _spelled(Gamma, regime, shape, rate)
    got = pl.DataFrame({"x": [x] * 4}).select(
        sf=dist.sf("x"), log_sf=dist.log_sf("x"), cdf=dist.cdf("x"), log_cdf=dist.log_cdf("x")
    )
    assert got["sf"][0] == 0.0
    assert got["log_sf"][0] == pytest.approx(log_sf, rel=1e-14, abs=0.0)
    assert got["cdf"][0] == 1.0
    assert got["log_cdf"][0] == 0.0


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
@pytest.mark.parametrize(
    ("shape", "rate", "x", "log_cdf"),
    [
        (2.5, 1.0, 1e-200, -1152.4935200993699),
        (50.0, 1.0, 1e-3, -493.8665112928519),
        (1e4, 1.0, 5e3, -1936.3029753315976),
    ],
    ids=["shape=2.5", "shape=50", "shape=1e4"],
)
def test_gamma_log_cdf_survives_the_lower_mass_underflowing(
    regime: Regime, shape: float, rate: float, x: float, log_cdf: float
) -> None:
    """`log_cdf` is the log prefactor plus the log of the series, finite where `cdf` is `e ** -1936`."""
    dist = _spelled(Gamma, regime, shape, rate)
    got = pl.DataFrame({"x": [x] * 4}).select(log_cdf=dist.log_cdf("x"), sf=dist.sf("x"))
    assert got["log_cdf"][0] == pytest.approx(log_cdf, rel=1e-14, abs=0.0)
    assert got["sf"][0] == 1.0


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
def test_gamma_log_cdf_keeps_a_point_whose_scaled_value_underflows(regime: Regime) -> None:
    """`Gamma(2, 0.3).log_cdf(1e-320)` is `2 ln(0.3e-320) - ln 2`, though `rate x` rounds to a subnormal.

    Below the normal range the point's log is `ln rate + ln x`, not `ln(rate x)`: the subnormal product has
    lost enough digits to move its log by `4.5e-7` relative. `pdf` on the same row is `rate ** 2 x` to one
    subnormal ulp.
    """
    x, rate = 1e-320, 0.3
    dist = _spelled(Gamma, regime, 2.0, rate)
    got = pl.DataFrame({"x": [x] * 4}).select(log_cdf=dist.log_cdf("x"), pdf=dist.pdf("x"))
    expected = 2.0 * (math.log(rate) + math.log(x)) - math.log(2.0)
    assert got["log_cdf"][0] == pytest.approx(expected, rel=1e-15, abs=0.0)
    assert got["pdf"][0] == pytest.approx(rate**2 * x, rel=0.0, abs=5e-324)


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
@pytest.mark.parametrize(
    ("shape", "x", "sf"), [(2.5, 715.0, 4.3468944051673152e-307), (0.5, 730.0, 1.9252843911342151e-319)], ids=str
)
def test_gamma_sf_answers_below_the_exp_underflow_threshold(regime: Regime, shape: float, x: float, sf: float) -> None:
    """`sf` stays nonzero down to the smallest subnormal, past the `e ** -709.78` where statrs returns `0`.

    A linear tail is the `exp` of its log, so its relative error is the log's absolute error: `1.3e-13` at
    `e ** -705`. The subnormal row holds to two of its ulps, all the precision a `1.9e-319` has.
    """
    dist = _spelled(Gamma, regime, shape, 1.0)
    got = pl.DataFrame({"x": [x] * 4}).select(sf=dist.sf("x"))
    assert got["sf"][0] == pytest.approx(sf, rel=1e-12, abs=2 * 5e-324)


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
def test_gamma_density_survives_a_power_leaving_float64(regime: Regime) -> None:
    """`Gamma(80, 1e-5).pdf(8e6)` is `4.46e-7`, where `rate ** shape` alone is `1e-400`.

    The density is the exponential of its log, so no single factor of
    `rate ** shape x ** (shape - 1) e ** (-rate x) / Gamma(shape)` is ever formed; statrs forms them
    and returns `NaN` here.
    """
    dist = _spelled(Gamma, regime, 80.0, 1e-5)
    got = pl.DataFrame({"x": [8e6] * 4}).select(pdf=dist.pdf("x"))
    assert got["pdf"][0] == pytest.approx(4.4556665770350952e-7, rel=1e-14, abs=0.0)


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
@pytest.mark.parametrize(
    ("shape", "pdf", "log_pdf"),
    [(0.5, math.inf, math.inf), (1.0, 4.0, math.log(4.0)), (1.5, 0.0, -math.inf)],
    ids=["shape<1", "shape=1", "shape>1"],
)
def test_gamma_density_at_the_origin_follows_the_shape(
    regime: Regime, shape: float, pdf: float, log_pdf: float
) -> None:
    """`pdf(0)` is infinite below `shape = 1`, `rate` at it and `0` above it, as `scipy.stats.gamma`."""
    dist = _spelled(Gamma, regime, shape, 4.0)
    got = pl.DataFrame({"x": [0.0] * 4}).select(pdf=dist.pdf("x"), log_pdf=dist.log_pdf("x"))
    assert got["pdf"][0] == pdf
    assert got["log_pdf"][0] == pytest.approx(log_pdf, rel=1e-15)


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
def test_gamma_log_pdf_keeps_the_linear_term_at_unit_shape(regime: Regime) -> None:
    """`Gamma(1, 1).log_pdf(1e-79)` is `-1e-79`: at `shape = 1` the log density is `-t` and nothing else.

    `ln Gamma(shape)` is `0` exactly there, and the `(shape - 1) ln t` term vanishes rather than
    cancelling a `shape ln t`, which would round the `-t` away.
    """
    x = 1e-79
    dist = _spelled(Gamma, regime, 1.0, 1.0)
    got = pl.DataFrame({"x": [x] * 4}).select(log_pdf=dist.log_pdf("x"))
    assert got["log_pdf"][0] == -x


# `(shape, rate, x, log_pdf, pdf)` at and just off `rate x = shape`, and one point in the bulk, below `shape = 0.2`.
_GAMMA_SMALL_SHAPE_DENSITY = [
    (1e-8, 1.0, 1e-8, -1.8843465087275504e-07, 0.9999998115653669),
    (1e-8, 1.0, 9.99999999e-09, -1.8743465088847987e-07, 0.99999981256536668),
    (1e-3, 2.0, 5e-4, 0.6858158188792466, 1.9853908943923007),
    (0.1, 1.0, 3.0, -6.241463711535505, 0.0019470035845404411),
]


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
@pytest.mark.parametrize(("shape", "rate", "x", "log_pdf", "pdf"), _GAMMA_SMALL_SHAPE_DENSITY, ids=lambda v: f"{v:.3g}")
def test_gamma_log_pdf_keeps_its_digits_at_a_small_shape(
    regime: Regime, shape: float, rate: float, x: float, log_pdf: float, pdf: float
) -> None:
    """`Gamma(1e-8, 1).log_pdf(1e-8)` is `-1.88e-7`, where `ln Gamma(shape)` and `-ln t` are both `18.4`.

    Below `shape = 0.2` the pair is read as `ln(shape / t)`, from the exact difference `shape - t` near
    `t = shape`: written term by term the two cancel to `2e-8` relative here, and a rounded `shape / t`
    alone leaves `5e-10` at `x = 9.99999999e-9`.
    """
    dist = _spelled(Gamma, regime, shape, rate)
    got = pl.DataFrame({"x": [x] * 4}).select(log_pdf=dist.log_pdf("x"), pdf=dist.pdf("x"))
    assert got["log_pdf"][0] == pytest.approx(log_pdf, rel=1e-14, abs=0.0)
    assert got["pdf"][0] == pytest.approx(pdf, rel=1e-15, abs=0.0)


# `(shape, rate, q, isf)`: the survival targets `ppf(1 - q)` cannot express at all, and one at a small shape.
_GAMMA_DEEP_ISF = [
    (2.0, 1.5, 1e-300, 464.88280758623506),
    (0.05, 1.0, 1e-300, 681.60702731455832),
    (100.0, 0.5, 1e-200, 1515.9084648225578),
]


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
@pytest.mark.parametrize(("shape", "rate", "q", "isf"), _GAMMA_DEEP_ISF, ids=lambda v: f"{v:.3g}")
def test_gamma_isf_resolves_a_survival_target_below_the_complement_floor(
    regime: Regime, shape: float, rate: float, q: float, isf: float
) -> None:
    """`isf(1e-300)` solves `ln Q(shape, t) = ln q` itself, where `1 - q` is exactly `1`."""
    dist = _spelled(Gamma, regime, shape, rate)
    got = pl.DataFrame({"q": [q] * 4}).select(isf=dist.isf("q"))
    assert got["isf"][0] == pytest.approx(isf, rel=1e-13, abs=0.0)


# `(shape, rate, q, ppf)`: the left tail, where the quantile is `(q Gamma(shape + 1)) ** (1 / shape) / rate`
# to leading order and a linear-scale Newton iteration on the cdf stalls on the vanishing density.
_GAMMA_DEEP_PPF = [
    (2.0, 1.5, 1e-300, 9.4280904158206338e-151),
    (1e-3, 1.0, 0.5, 5.2442064082779784e-302),
]


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
@pytest.mark.parametrize(("shape", "rate", "q", "ppf"), _GAMMA_DEEP_PPF, ids=lambda v: f"{v:.3g}")
def test_gamma_ppf_keeps_relative_precision_in_the_left_tail(
    regime: Regime, shape: float, rate: float, q: float, ppf: float
) -> None:
    """The inverse iterates on `ln t`, so a quantile of `5e-302` keeps relative precision.

    `1e-13` rather than tighter: the log-scale solve carries `|ln t| * 1.1e-16` relative, `7.7e-14` at
    `t = 5e-302`.
    """
    dist = _spelled(Gamma, regime, shape, rate)
    got = pl.DataFrame({"q": [q] * 4}).select(ppf=dist.ppf("q"))
    assert got["ppf"][0] == pytest.approx(ppf, rel=1e-13, abs=0.0)


# `(shape, entropy)` at `rate = 2.5`, straddling the switch to the asymptotic series at `shape = 30`.
_GAMMA_ENTROPY = [
    (29.9, 2.1903351514886984),
    (29.999, 2.1920253438735657),
    (30.0, 2.1920423874136325),
    (30.1, 2.1937438146103469),
    (1e4, 5.1077846531519313),
    (1e8, 9.7129881699733671),
]


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
@pytest.mark.parametrize(("shape", "entropy"), _GAMMA_ENTROPY, ids=lambda v: f"{v:.4g}")
def test_gamma_entropy_keeps_its_digits_at_a_large_shape(regime: Regime, shape: float, entropy: float) -> None:
    """`Gamma(1e8, 2.5).entropy()` holds `1e-13` relative, where statrs' closed form is `2.3e-8` off.

    `shape + ln Gamma(shape) + (1 - shape) psi(shape)` cancels terms of order `shape ln shape` down to
    `ln(shape) / 2`; from `shape = 30` the entropy is its asymptotic series, and below it the closed form
    holds `1e-13`.
    """
    dist = _spelled(Gamma, regime, shape, 2.5)
    got = pl.DataFrame({"_": range(4)}).select(entropy=dist.entropy())
    assert got["entropy"][0] == pytest.approx(entropy, rel=1e-13, abs=0.0)


def test_gamma_holds_the_bulk_at_a_shape_past_any_series_budget() -> None:
    """At `shape = 2e12` the bulk is Temme's expansion, where the series would need `~1.3e7` terms.

    `P(shape, shape) = 1/2 + 1 / (3 sqrt(2 pi shape))` and the median is `shape - 1/3 + 8 / (405 shape)`,
    both to `O(shape ** -3/2)`: the median holds `5e-16` relative, inside the `|ln t| * 1.1e-16` the log-scale
    inverse carries.
    """
    shape = 2e12
    dist = Gamma(shape=shape, rate=1.0)
    got = pl.DataFrame({"x": [shape]}).select(cdf=dist.cdf("x"), median=dist.median())
    assert got["cdf"][0] == pytest.approx(0.5 + 1.0 / (3.0 * math.sqrt(2.0 * math.pi * shape)), rel=1e-14, abs=0.0)
    assert got["median"][0] == pytest.approx(shape - 1.0 / 3.0 + 8.0 / (405.0 * shape), rel=1e-15, abs=0.0)


# `(seam, shape, x, log_cdf, log_sf)` on both sides of every switch between the incomplete gamma's regions,
# at `rate = 1` so `t = x`: Cephes' split at `t = 0.5`, `t = 1.1` and `shape ln t = -0.4`, the Stirling
# prefactor from `shape = 10`, Temme's expansion from `shape = 100` and inside `0.3 shape` of the mean, and
# its `erfcx` switching from a Taylor polynomial to a continued fraction at `w = 2`.
_GAMMA_SEAMS = [
    ("t=0.5", 0.56, 0.49999999999999994, -0.43979002563659664, -1.0333068734729631),
    ("t=0.5", 0.56, 0.5, -0.4397900256365966, -1.0333068734729631),
    ("t=0.5", 0.56, 0.5000000000000001, -0.43979002563659647, -1.0333068734729633),
    ("t=1.1", 1.15, 1.0999999999999999, -0.5012941373879154, -0.9307604993385232),
    ("t=1.1", 1.15, 1.1, -0.5012941373879153, -0.9307604993385235),
    ("t=1.1", 1.15, 1.1000000000000003, -0.5012941373879152, -0.9307604993385237),
    ("shape*ln(t)=-0.4", 0.4, 0.3678794411714423, -0.37992156645103786, -1.1517442692242488),
    ("shape*ln(t)=-0.4", 0.4, 0.36787944117144233, -0.3799215664510378, -1.1517442692242488),
    ("shape*ln(t)=-0.4", 0.4, 0.3678794411714424, -0.37992156645103775, -1.151744269224249),
    ("shape=10", 9.999999999999998, 10.0, -0.6123596078537543, -0.7810395684962785),
    ("shape=10", 9.999999999999998, 25.0, -0.0002215011678213132, -8.415193444701625),
    ("shape=10", 10.0, 10.0, -0.6123596078537547, -0.781039568496278),
    ("shape=10", 10.0, 25.0, -0.0002215011678213136, -8.415193444701623),
    ("shape=100", 99.99999999999999, 110.0, -0.17230628158645353, -1.8433980649510182),
    ("shape=100", 100.0, 110.0, -0.17230628158645392, -1.843398064951016),
    ("|t-shape|=0.3*shape", 1000.0, 699.9999999999999, -59.85147851292529, -1.0158583345332718e-26),
    ("|t-shape|=0.3*shape", 1000.0, 700.0, -59.85147851292524, -1.0158583345333217e-26),
    ("|t-shape|=0.3*shape", 1000.0, 700.0000000000001, -59.85147851292519, -1.0158583345333715e-26),
    ("|t-shape|=0.3*shape", 1000.0, 1299.9999999999998, -1.8736155715786546e-18, -40.81866164901851),
    ("|t-shape|=0.3*shape", 1000.0, 1300.0, -1.8736155715785553e-18, -40.818661649018566),
    ("|t-shape|=0.3*shape", 1000.0, 1300.0000000000002, -1.8736155715784555e-18, -40.818661649018615),
    ("erfcx w=2", 1000.0, 1092.1290262434488, -0.0022661064301416566, -6.09082498843765),
    ("erfcx w=2", 1000.0, 1092.129026243449, -0.0022661064301416085, -6.090824988437672),
    ("erfcx w=2", 1000.0, 1092.1290262434493, -0.0022661064301415603, -6.090824988437693),
]

# `(shape, x, log_pdf)` either side of the density's switches at `shape = 0.2` and `shape = 10`.
_GAMMA_DENSITY_SEAMS = [
    (0.19999999999999998, 0.2, -0.43651349248350435),
    (0.2, 0.2, -0.43651349248350424),
    (0.19999999999999998, 1.5, -3.348435908917316),
    (0.2, 1.5, -3.348435908917316),
    (9.999999999999998, 10.0, -2.0785616431350586),
    (10.0, 10.0, -2.0785616431350586),
]

# Below `shape = 10` the prefactor's `ln Gamma(shape + 1)` is statrs', whose `~1e-14` absolute error the logs
# carry: `2.4e-14` relative on `log_cdf` at the mean of `Gamma(9.999999999999998, 1)`, `1.6e-15` at `shape = 10`.
_GAMMA_SEAM_RTOL = 3e-14


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
@pytest.mark.parametrize(
    ("shape", "x", "log_cdf", "log_sf"), [pytest.param(*row, id=f"{seam}:x={row[1]!r}") for seam, *row in _GAMMA_SEAMS]
)
def test_gamma_tails_hold_on_both_sides_of_every_region_seam(
    regime: Regime, shape: float, x: float, log_cdf: float, log_sf: float
) -> None:
    """Both logs match mpmath at a seam, one ulp below it and one above it, so a branch that breaks shows up here."""
    dist = _spelled(Gamma, regime, shape, 1.0)
    got = pl.DataFrame({"x": [x] * 4}).select(log_cdf=dist.log_cdf("x"), log_sf=dist.log_sf("x"))
    assert got["log_cdf"][0] == pytest.approx(log_cdf, rel=_GAMMA_SEAM_RTOL, abs=0.0)
    assert got["log_sf"][0] == pytest.approx(log_sf, rel=_GAMMA_SEAM_RTOL, abs=0.0)


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
@pytest.mark.parametrize(("shape", "x", "log_pdf"), _GAMMA_DENSITY_SEAMS, ids=repr)
def test_gamma_log_pdf_holds_on_both_sides_of_its_shape_seams(
    regime: Regime, shape: float, x: float, log_pdf: float
) -> None:
    """`log_pdf` matches mpmath one ulp either side of `shape = 0.2` and `shape = 10`."""
    dist = _spelled(Gamma, regime, shape, 1.0)
    got = pl.DataFrame({"x": [x] * 4}).select(log_pdf=dist.log_pdf("x"))
    assert got["log_pdf"][0] == pytest.approx(log_pdf, rel=_GAMMA_SEAM_RTOL, abs=0.0)
