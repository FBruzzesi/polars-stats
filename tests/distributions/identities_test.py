"""Algebraic identities: one distribution reducing to another, and a distribution's own symmetry.

None of these is a numeric-parity claim, so scipy is not the oracle: the *other side of the identity*
is. They are per distribution by nature and would be sentinels eight rows out of nine in the registry,
so they live here rather than as fields.

Its neighbour is [`precision_test.py`](precision_test.py), which owns the numerical-regime facts with
no oracle at all.
"""

from __future__ import annotations

import math

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from polars_stats import Bernoulli, Beta, Binomial, Exponential, Geometric, Normal, Uniform
from tests._polars_compat import assert_series_equal

_NEG_INF = float("-inf")

# --------------------------------------------------------------------------------------------------
# Beta(1, 1) is Uniform(0, 1).
# --------------------------------------------------------------------------------------------------

_UNIT_XS = [-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5]  # interior, both boundaries, and both sides outside
_UNIT_QS = [0.0, 0.1, 0.5, 0.9, 1.0]

_UNIT_BETA = Beta(a=1.0, b=1.0)
_UNIT_UNIFORM = Uniform(min=0.0, max=1.0)


@pytest.mark.parametrize("method", ["pdf", "log_pdf", "cdf", "log_cdf", "sf", "log_sf"])
def test_beta_one_one_value_keyed_methods_match_the_unit_uniform(method: str) -> None:
    df = pl.DataFrame({"x": _UNIT_XS})
    via_beta = df.select(r=getattr(_UNIT_BETA, method)(pl.col("x")))["r"]
    via_uniform = df.select(r=getattr(_UNIT_UNIFORM, method)(pl.col("x")))["r"]
    assert_series_equal(via_beta, via_uniform, rel_tol=0.0, abs_tol=1e-12)


@pytest.mark.parametrize("method", ["ppf", "isf"])
def test_beta_one_one_quantile_methods_match_the_unit_uniform(method: str) -> None:
    # The inverse regularized incomplete beta is Newton-refined, so agreement is close but not exact.
    df = pl.DataFrame({"q": _UNIT_QS})
    via_beta = df.select(r=getattr(_UNIT_BETA, method)(pl.col("q")))["r"]
    via_uniform = df.select(r=getattr(_UNIT_UNIFORM, method)(pl.col("q")))["r"]
    assert_series_equal(via_beta, via_uniform, rel_tol=0.0, abs_tol=1e-9)


@pytest.mark.parametrize("method", ["mean", "variance", "std", "median", "entropy"])
def test_beta_one_one_moments_match_the_unit_uniform(method: str, single_row_frame: pl.DataFrame) -> None:
    via_beta = single_row_frame.select(r=getattr(_UNIT_BETA, method)()).item(0, "r")
    via_uniform = single_row_frame.select(r=getattr(_UNIT_UNIFORM, method)()).item(0, "r")
    assert via_beta == pytest.approx(via_uniform, abs=1e-9)


# --------------------------------------------------------------------------------------------------
# Beta(a, b) reflects into Beta(b, a): `F_{a,b}(x) = 1 - F_{b,a}(1 - x)`.
# --------------------------------------------------------------------------------------------------

_BETA_SHAPES = [(2.0, 3.0), (0.5, 0.5), (1.0, 1.0), (5.0, 1.0), (2.0, 8.0)]
_BETA_GRID = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]


@pytest.mark.parametrize(("a", "b"), _BETA_SHAPES, ids=[f"a={a},b={b}" for a, b in _BETA_SHAPES])
def test_beta_cdf_has_the_reflection_symmetry(a: float, b: float) -> None:
    """`X ~ Beta(a, b)` implies `1 - X ~ Beta(b, a)`."""
    df = pl.DataFrame({"x": _BETA_GRID})
    lhs = df.select(r=Beta(a=a, b=b).cdf(pl.col("x")))["r"]
    rhs = df.select(r=1 - Beta(a=b, b=a).cdf(1 - pl.col("x")))["r"]
    assert_series_equal(lhs, rhs, rel_tol=0.0, abs_tol=1e-12)


# --------------------------------------------------------------------------------------------------
# Binomial(n=1, p) is Bernoulli(p).
# --------------------------------------------------------------------------------------------------

_PROBS = [0.0, 0.2, 0.5, 0.8, 1.0]
_BERNOULLI_GRID = [-1.0, 0.0, 0.5, 1.0, 2.0]
_BERNOULLI_QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]


@pytest.mark.parametrize("p", _PROBS)
@pytest.mark.parametrize("method", ["pmf", "log_pmf", "cdf", "log_cdf", "sf", "log_sf"])
def test_single_trial_binomial_value_keyed_methods_match_bernoulli(p: float, method: str) -> None:
    df = pl.DataFrame({"x": _BERNOULLI_GRID})
    binom = df.select(r=getattr(Binomial(1, p), method)(pl.col("x")))["r"]
    bern = df.select(r=getattr(Bernoulli(p), method)(pl.col("x")))["r"]
    assert_series_equal(binom, bern, check_names=False)


@pytest.mark.parametrize("p", _PROBS)
@pytest.mark.parametrize("method", ["mean", "variance", "std", "entropy"])
def test_single_trial_binomial_moments_match_bernoulli(p: float, method: str, single_row_frame: pl.DataFrame) -> None:
    binom = single_row_frame.select(r=getattr(Binomial(1, p), method)()).item(0, "r")
    bern = single_row_frame.select(r=getattr(Bernoulli(p), method)()).item(0, "r")
    assert binom == pytest.approx(bern)


@pytest.mark.parametrize("p", [0.2, 0.5, 0.8])
@pytest.mark.parametrize("q", _BERNOULLI_QUANTILES)
def test_single_trial_binomial_ppf_matches_bernoulli(p: float, q: float, single_row_frame: pl.DataFrame) -> None:
    binom = single_row_frame.select(r=Binomial(1, p).ppf(q)).item(0, "r")
    bern = single_row_frame.select(r=Bernoulli(p).ppf(q)).item(0, "r")
    assert binom == bern


# --------------------------------------------------------------------------------------------------
# The exponential is memoryless, and the normal density is symmetric about its mean.
# --------------------------------------------------------------------------------------------------


@given(
    rate=st.floats(min_value=1e-2, max_value=5.0),
    s=st.floats(min_value=0.0, max_value=15.0),
    t=st.floats(min_value=0.0, max_value=15.0),
)
def test_the_exponential_is_memoryless(rate: float, s: float, t: float) -> None:
    """The exponential's defining property: `P(X > s + t | X > s) = P(X > t)`.

    `P(X > s + t | X > s) = sf(s + t) / sf(s)`, which must equal `sf(t)` for every `s, t >= 0`. The
    parameter and offset ranges are bounded so `rate * (s + t)` stays well clear of float underflow,
    keeping both survival values strictly positive (no `0 / 0`).
    """
    dist = Exponential(rate=rate)
    df = pl.DataFrame({"_": [0]})

    conditional = df.select(r=dist.sf(s + t) / dist.sf(s))["r"]
    marginal = df.select(r=dist.sf(t))["r"]

    assert_series_equal(conditional, marginal, rel_tol=1e-9, abs_tol=1e-15, check_names=False)


def test_the_normal_density_is_symmetric_about_its_mean() -> None:
    """`pdf(mu - d) == pdf(mu + d)` exactly, which the parity grid implies only within its tolerance."""
    mean, std = 2.0, 1.5
    df = pl.DataFrame({"d": [0.5, 1.0, 4.0]})
    dist = Normal(mu=mean, sigma=std)
    left = df.select(r=dist.pdf(mean - pl.col("d")))["r"]
    right = df.select(r=dist.pdf(mean + pl.col("d")))["r"]
    assert_series_equal(left, right)


# --------------------------------------------------------------------------------------------------
# Geometric at `p = 1`: the whole mass on `k = 1`, and the branch ordering that keeps `NaN` out of it.
#
# At `p = 1` the shared entry `log1p(-p)` is `-inf`, so `floor(value) * log1p(-p)` is `NaN` for every
# `value` in `[0, 1)` and `-inf` above it. Each of `_cdf`, `_log_cdf`, `_sf` and `_log_sf` answers its
# below-support constant *before* forming that product, and each inverse short-circuits on `p == 1`
# before dividing `-inf` by `-inf`. Nothing else pins that ordering: the scipy parity grid excludes
# `p = 1` (scipy's generic discrete `ppf`, `isf`, `median` and `entropy` do not answer there) and the
# registry's own strategy tops out below it, so a reordered branch would leak `NaN` and still ship green.
# --------------------------------------------------------------------------------------------------

# One row per (method, value) with the exact answer of the `p = 1` point mass. Keyed by method name
# and reached with `getattr`, as the rest of this file does: a lambda per row would only respell the
# key it is already filed under.
_MASS_POINT_CASES: dict[str, list[tuple[float, float]]] = {
    "pmf": [(-1.0, 0.0), (0.0, 0.0), (0.5, 0.0), (1.0, 1.0), (2.0, 0.0), (10.0, 0.0)],
    "log_pmf": [(-1.0, _NEG_INF), (0.0, _NEG_INF), (0.5, _NEG_INF), (1.0, 0.0), (2.0, _NEG_INF)],
    "cdf": [(-1.0, 0.0), (0.0, 0.0), (0.5, 0.0), (1.0, 1.0), (2.0, 1.0), (1e6, 1.0)],
    "log_cdf": [(-1.0, _NEG_INF), (0.0, _NEG_INF), (0.5, _NEG_INF), (1.0, 0.0), (2.0, 0.0), (1e6, 0.0)],
    "sf": [(-1.0, 1.0), (0.0, 1.0), (0.5, 1.0), (1.0, 0.0), (2.0, 0.0), (1e6, 0.0)],
    "log_sf": [(-1.0, 0.0), (0.0, 0.0), (0.5, 0.0), (1.0, _NEG_INF), (2.0, _NEG_INF), (1e6, _NEG_INF)],
}


@pytest.mark.parametrize(("method", "cases"), _MASS_POINT_CASES.items())
def test_degenerate_geometric_value_keyed_method(
    method: str, cases: list[tuple[float, float]], single_row_frame: pl.DataFrame
) -> None:
    """Exact, and never `NaN`: `0 * -inf` must not reach the result on either side of the support."""
    for value, expected in cases:
        got = single_row_frame.select(r=getattr(Geometric(p=1.0), method)(pl.lit(value)))["r"].item()
        assert not math.isnan(got), f"{method}({value}) -> NaN"
        assert got == expected, f"{method}({value}) -> {got}, want {expected}"


@pytest.mark.parametrize(("method", "cases"), _MASS_POINT_CASES.items())
def test_degenerate_geometric_value_keyed_method_from_a_column(method: str, cases: list[tuple[float, float]]) -> None:
    """The per-row path answers identically: `p = 1` is a value, not a constant-folding artefact."""
    frame = pl.DataFrame({"p": [1.0] * len(cases), "v": [v for v, _ in cases]})
    got = frame.select(r=getattr(Geometric(p=pl.col("p")), method)(pl.col("v")))["r"].to_list()
    assert got == [e for _, e in cases]


_DEGENERATE_QUANTILES = [0.0, 1e-300, 0.25, 0.5, 0.75, 1.0]


@pytest.mark.parametrize("quantile", _DEGENERATE_QUANTILES)
def test_degenerate_geometric_inverses_collapse_to_the_mass_point(
    quantile: float, single_row_frame: pl.DataFrame
) -> None:
    """Every quantile inverts to `k = 1`, including the endpoints where the ratio would be `NaN`."""
    frame = single_row_frame.select(ppf=Geometric(p=1.0).ppf(quantile), isf=Geometric(p=1.0).isf(quantile))
    assert frame.item(0, "ppf") == 1.0
    assert frame.item(0, "isf") == 1.0


@pytest.mark.parametrize("method", ["ppf", "isf"])
def test_degenerate_geometric_inverses_collapse_to_the_mass_point_from_a_column(method: str) -> None:
    """The per-row path short-circuits on `p == 1` too: `p = 1` is a value, not a constant-folding artefact."""
    frame = pl.DataFrame({"p": [1.0] * len(_DEGENERATE_QUANTILES), "q": _DEGENERATE_QUANTILES})
    got = frame.select(r=getattr(Geometric(p=pl.col("p")), method)(pl.col("q")))["r"].to_list()
    assert got == [1.0] * len(_DEGENERATE_QUANTILES)


def test_degenerate_geometric_moments(single_row_frame: pl.DataFrame) -> None:
    """A point mass at `k = 1`: mean `1`, no spread, and entropy `0` by the `0 log 0 = 0` convention."""
    g = Geometric(p=1.0)
    got = single_row_frame.select(
        mean=g.mean(), variance=g.variance(), std=g.std(), median=g.median(), entropy=g.entropy()
    )
    assert got.item(0, "mean") == 1.0
    assert got.item(0, "variance") == 0.0
    assert got.item(0, "std") == 0.0
    assert got.item(0, "median") == 1.0
    assert got.item(0, "entropy") == 0.0


def test_degenerate_geometric_sample_is_always_the_mass_point() -> None:
    """Every trial succeeds, so every draw is `1`."""
    frame = pl.DataFrame({"_": range(64)})
    drawn = frame.select(r=Geometric(p=1.0).sample(seed=7))["r"]
    assert drawn.dtype == pl.UInt64
    assert drawn.unique().to_list() == [1]
