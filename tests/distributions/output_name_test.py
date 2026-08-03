"""Output-name contract of `sample` / `samples` and the value-keyed methods.

With all-constant parameters there is no input column to inherit a name from, so the samplers get
the deliberate default names `"sample"` / `"samples"` (rather than leaking an internal expression
name). With column-valued parameters the output keeps polars root-name semantics: it is named after
the first parameter expression, which is what lets multi-column parameters (`pl.col("p1", "p2")`)
and `.name.*` modifiers work.

Value-keyed methods are named after the evaluation column: `propagate_null_and_nan` re-aliases its
output to `value`'s resolved name, since its leading `pl.lit(None)` branch would otherwise name
every value-keyed output `"literal"`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from polars_stats import Bernoulli, Beta, Binomial, Exponential, LogNormal, Normal, Uniform

if TYPE_CHECKING:
    from polars_stats.distributions._base import _UnivariateDistribution

_FRAME = pl.DataFrame({"p": [0.5, 0.5], "mu": [0.0, 0.0], "n": [5, 5]})

_SCALAR_PARAMS: dict[str, _UnivariateDistribution] = {
    "bernoulli": Bernoulli(p=0.5),
    "binomial": Binomial(n=5, p=0.5),
    "normal": Normal(mu=0.0, sigma=1.0),
    "lognormal": LogNormal(mu=0.0, sigma=1.0),
    "uniform": Uniform(min=0.0, max=1.0),
    "exponential": Exponential(rate=1.0),
    "beta": Beta(a=2.0, b=3.0),
}

# Distribution with a column-valued first parameter, paired with the root name the output inherits.
_COLUMN_PARAMS: dict[str, tuple[_UnivariateDistribution, str]] = {
    "bernoulli": (Bernoulli(p=pl.col("p")), "p"),
    "binomial": (Binomial(n=pl.col("n"), p=0.5), "n"),
    "normal": (Normal(mu=pl.col("mu"), sigma=1.0), "mu"),
    "lognormal": (LogNormal(mu=pl.col("mu"), sigma=1.0), "mu"),
    "uniform": (Uniform(min=pl.col("mu"), max=1.0), "mu"),
    # `mu` is all-zero (an invalid rate); the rate column reads the positive `p` column instead.
    "exponential": (Exponential(rate=pl.col("p")), "p"),
    # Same for the first shape: `p` is positive on every row, a valid `a`.
    "beta": (Beta(a=pl.col("p"), b=1.0), "p"),
}


@pytest.mark.parametrize("dist", _SCALAR_PARAMS.values(), ids=list(_SCALAR_PARAMS))
def test_sample_with_scalar_params_is_named_sample(dist: _UnivariateDistribution) -> None:
    assert _FRAME.select(dist.sample(seed=0)).columns == ["sample"]


@pytest.mark.parametrize("dist", _SCALAR_PARAMS.values(), ids=list(_SCALAR_PARAMS))
def test_samples_with_scalar_params_is_named_samples(dist: _UnivariateDistribution) -> None:
    assert _FRAME.select(dist.samples(size=3, seed=0)).columns == ["samples"]


@pytest.mark.parametrize(("dist", "root"), _COLUMN_PARAMS.values(), ids=list(_COLUMN_PARAMS))
def test_sample_with_column_params_keeps_root_name(dist: _UnivariateDistribution, root: str) -> None:
    assert _FRAME.select(dist.sample(seed=0)).columns == [root]


@pytest.mark.parametrize(("dist", "root"), _COLUMN_PARAMS.values(), ids=list(_COLUMN_PARAMS))
def test_samples_with_column_params_keeps_root_name(dist: _UnivariateDistribution, root: str) -> None:
    assert _FRAME.select(dist.samples(size=3, seed=0)).columns == [root]


@pytest.mark.parametrize("dist", _SCALAR_PARAMS.values(), ids=list(_SCALAR_PARAMS))
def test_value_keyed_keeps_value_root_name(dist: _UnivariateDistribution) -> None:
    """`cdf(pl.col("p"))` is named `"p"` via the guard's alias.

    Only the output name is pinned: how a downstream `.name.*` modifier resolves it is polars'
    call and differs across supported versions (old polars resolves the *root* name, which for the
    closed-form hooks is the `pl.len()` inside a scalar parameter's `pl.repeat` expansion).
    """
    assert _FRAME.select(dist.cdf(pl.col("p"))).columns == ["p"]
