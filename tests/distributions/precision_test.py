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

import polars as pl
import pytest
from scipy.stats import binom as scipy_binom

from polars_stats import Beta, Binomial, DiscreteUniform, Exponential, Geometric

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
