"""Every value-keyed method reports an invalid parameterisation, whichever branch the value selects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl
import pytest

from polars_stats import (
    Bernoulli,
    Beta,
    Binomial,
    DiscreteUniform,
    Exponential,
    Geometric,
    LogNormal,
    Normal,
    Uniform,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from polars_stats.distributions._base import _UnivariateDistribution


@dataclass(frozen=True)
class _Case:
    """One distribution parameterised from columns, where row 0 is valid and row 1 is not.

    `fragment` is matched against the raised message, so it names the offending parameter.
    """

    name: str
    dist: _UnivariateDistribution
    columns: dict[str, list[float]]
    fragment: str


_CASES: tuple[_Case, ...] = (
    _Case("Bernoulli", Bernoulli(p=pl.col("p")), {"p": [0.3, 1.5]}, "p"),
    _Case("Beta", Beta(a=pl.col("a"), b=pl.col("b")), {"a": [2.0, -1.0], "b": [3.0, 3.0]}, "a"),
    _Case("Binomial", Binomial(n=pl.col("n"), p=pl.col("p")), {"n": [10, 10], "p": [0.3, 1.5]}, "p"),
    _Case("DiscreteUniform", DiscreteUniform(min=pl.col("lo"), max=pl.col("hi")), {"lo": [0, 5], "hi": [10, 2]}, "max"),
    _Case("Exponential", Exponential(rate=pl.col("r")), {"r": [1.0, -1.0]}, "rate"),
    _Case("Geometric", Geometric(p=pl.col("p")), {"p": [0.3, 1.5]}, "p"),
    _Case("LogNormal", LogNormal(mu=pl.col("m"), sigma=pl.col("s")), {"m": [0.0, 0.0], "s": [1.0, -1.0]}, "sigma"),
    _Case("Normal", Normal(mu=pl.col("m"), sigma=pl.col("s")), {"m": [0.0, 0.0], "s": [1.0, -1.0]}, "sigma"),
    _Case("Uniform", Uniform(min=pl.col("lo"), max=pl.col("hi")), {"lo": [0.0, 5.0], "hi": [1.0, 2.0]}, "max"),
)

_VALUE_METHODS = ("pdf", "log_pdf", "pmf", "log_pmf", "cdf", "log_cdf", "sf", "log_sf")
"""Support-point methods. `pdf` / `pmf` are family-specific, so a missing one is skipped, not failed."""

_QUANTILE_METHODS = ("ppf", "isf")

_SUPPORT_POINTS = (-3.0, -1.0, 0.0, 0.5, 1.0, 3.0, 100.0)
"""Spans below, inside and above every support, so each value-keyed branch gets at least one value."""

_QUANTILES = (0.001, 0.5, 0.999)


def _ids(case: _Case) -> str:
    return case.name


@pytest.mark.parametrize("case", _CASES, ids=_ids)
@pytest.mark.parametrize(
    ("method", "value"),
    [(m, v) for m in _VALUE_METHODS for v in _SUPPORT_POINTS] + [(m, q) for m in _QUANTILE_METHODS for q in _QUANTILES],
)
def test_invalid_parameter_raises_whichever_branch_the_value_selects(case: _Case, method: str, value: float) -> None:
    method_fn: Callable[[pl.Expr], pl.Expr] | None = getattr(case.dist, method, None)
    if method_fn is None:
        pytest.skip(f"{case.name} has no {method} (wrong family)")

    frame = pl.DataFrame({**case.columns, "v": [value, value]})
    # A `DID NOT RAISE` here means the invalid row in `columns` was accepted, which is what the
    # `polars<1.44.0` cap in `pyproject.toml` exists to prevent.
    with pytest.raises(pl.exceptions.ComputeError, match=case.fragment):
        frame.select(r=method_fn(pl.col("v")))
