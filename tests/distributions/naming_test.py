"""Output names, and what a column-name `str` means in either argument position.

Two contracts that only look separate. Polars resolves a plugin expression's output name from its
*first* input, so both come down to which input a call puts first: the samplers have no input column
with all-constant parameters and take a deliberate default name, and everything else inherits from
the evaluation point or the first parameter.

Every case is discriminating because no two parameters of one distribution read the same column
(`registry_test.test_spec_parameter_columns_are_distinct_and_in_the_contract_frame`): an expression
that followed `inputs[1]` instead of the first-input rule would inherit the wrong name and fail.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.testing import assert_series_equal

from tests._registry import VALUE_KEYED_METHODS, all_value_keyed, point_column

if TYPE_CHECKING:
    from collections.abc import Callable

    from polars_stats.distributions._base import _UnivariateDistribution
    from tests._registry import DistSpec


# The output-name half.


def test_sample_with_scalar_params_is_named_sample(dist: _UnivariateDistribution, contract_frame: pl.DataFrame) -> None:
    assert contract_frame.select(dist.sample(seed=0)).columns == ["sample"]


def test_samples_with_scalar_params_is_named_samples(
    dist: _UnivariateDistribution, contract_frame: pl.DataFrame
) -> None:
    assert contract_frame.select(dist.samples(size=3, seed=0)).columns == ["samples"]


def test_sample_with_column_params_keeps_root_name(spec: DistSpec, contract_frame: pl.DataFrame) -> None:
    root = spec.parameters[0].column
    assert contract_frame.select(spec.build("name").sample(seed=0)).columns == [root]


def test_samples_with_column_params_keeps_root_name(spec: DistSpec, contract_frame: pl.DataFrame) -> None:
    root = spec.parameters[0].column
    assert contract_frame.select(spec.build("name").samples(size=3, seed=0)).columns == [root]


@pytest.mark.parametrize("method", VALUE_KEYED_METHODS)
def test_value_keyed_keeps_value_root_name(
    dist: _UnivariateDistribution, method: str, contract_frame: pl.DataFrame
) -> None:
    """`cdf(pl.col("q"))` is named `"q"`.

    Only the output name is pinned: how a downstream `.name.*` modifier resolves it is polars'
    call and differs across supported versions. `q` is the one column every method accepts, since
    `ppf` / `isf` read a quantile.
    """
    assert contract_frame.select(getattr(dist, method)(pl.col("q"))).columns == ["q"]


def test_validator_with_column_params_keeps_first_parameter_root_name(
    spec: DistSpec, contract_frame: pl.DataFrame
) -> None:
    """`_validated_params` is the unaliased read.

    The distributions routed through `_moment` wrap it in a `pl.when(...).then(value)` gate that
    would rename the output and make any assertion pass, so the validator is reached directly.
    """
    root = spec.parameters[0].column
    assert contract_frame.select(spec.build("name")._validated_params).columns == [root]


# The `str` argument half.


def test_str_value_arg_equals_col_expr(spec: DistSpec, contract_frame: pl.DataFrame) -> None:
    """Every value-keyed method takes its evaluation point as a column name.

    Looped over the methods rather than parametrised: the claim is one per distribution, and the frame
    and the distribution are the same for every method.
    """
    dist = spec.build("name")
    for method in all_value_keyed(dist):
        column = point_column(method)
        via_str = contract_frame.select(r=getattr(dist, method)(column))["r"]
        via_expr = contract_frame.select(r=getattr(dist, method)(pl.col(column)))["r"]
        assert_series_equal(via_str, via_expr)
        assert via_str.null_count() < via_str.len(), f"{method} answered all-null"


_SAMPLERS: list[Callable[[_UnivariateDistribution], pl.Expr]] = [
    lambda d: d.sample(seed=0),
    lambda d: d.samples(size=3, seed=0),
]


@pytest.mark.parametrize("call", _SAMPLERS, ids=["sample", "samples"])
def test_str_parameter_equals_col_expr(
    spec: DistSpec, call: Callable[[_UnivariateDistribution], pl.Expr], contract_frame: pl.DataFrame
) -> None:
    """A parameter spelled as a column name draws the same rows as the same parameter as `pl.col`."""
    by_name = spec.build("name")
    by_expr = spec.cls(**{p.name: pl.col(p.column) for p in spec.parameters})

    got = contract_frame.select(r=call(by_name))["r"]
    want = contract_frame.select(r=call(by_expr))["r"]
    assert_series_equal(got, want)
    assert got.null_count() == 0
