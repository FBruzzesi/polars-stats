"""Call-time contract of the two sampler kwargs, `seed` and `size`."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl
import pytest
from polars.exceptions import ComputeError

import polars_stats as ps
from polars_stats.distributions._base import _MAX_WIRE_INT
from tests.property._specs import ALL_SPECS

if TYPE_CHECKING:
    from collections.abc import Callable

    from polars_stats.distributions._base import _UnivariateDistribution
    from tests.property._specs import DistSpec

    SamplingCallable = Callable[[_UnivariateDistribution, int | None], pl.Expr]

_ROWS = 3
_FRAME = pl.DataFrame({"_": range(_ROWS)})
_SIZE = 2

_SEED_RANGE_MESSAGE = r"seed must be in \[0, 2\*\*63\), got"
_SEED_TYPE_MESSAGE = r"seed should be an int or None, found "

_OUT_OF_RANGE = [-1, -(2**70), _MAX_WIRE_INT + 1, 2**64 - 1, 2**64, 2**70]
"""A negative seed and one past the `i64` ceiling, since the guard has to catch both directions."""

_NON_INTEGERS = [1.0, 1.5, True, "7", np.int64(5)]
"""Every non-`int` a `seed` or `size` arrives as, including `True` and `np.int64(5)`, which pass a range check."""


_CALLS: dict[str, SamplingCallable] = {
    "sample": lambda dist, seed: dist.sample(seed=seed),
    "samples": lambda dist, seed: dist.samples(size=_SIZE, seed=seed),
}
"""The two public samplers, the only two entry points that take a seed."""


@pytest.mark.parametrize("call", _CALLS.values(), ids=_CALLS.keys())
@pytest.mark.parametrize("path", ["scalar", "column"])
@pytest.mark.parametrize("seed", _OUT_OF_RANGE)
def test_an_out_of_range_seed_raises_at_call_time(seed: int, path: str, call: SamplingCallable) -> None:
    """`ValueError` from building the expression, before any frame is touched."""
    dist = ps.Normal(0.0, 1.0) if path == "scalar" else ps.Normal("m", "s")

    with pytest.raises(ValueError, match=_SEED_RANGE_MESSAGE):
        call(dist, seed)


@pytest.mark.parametrize("call", _CALLS.values(), ids=_CALLS.keys())
@pytest.mark.parametrize("path", ["scalar", "column"])
@pytest.mark.parametrize("seed", _NON_INTEGERS, ids=[type(seed).__name__ for seed in _NON_INTEGERS])
def test_a_non_integer_seed_raises_at_call_time(seed: object, path: str, call: SamplingCallable) -> None:
    """`TypeError`, not the `ValueError` an out-of-range seed gets, and not a serde error at collect."""
    dist = ps.Normal(0.0, 1.0) if path == "scalar" else ps.Normal("m", "s")

    # `seed: int | None` is what the type checkers enforce; the cast is how this test reaches the
    # callers who get past them.
    with pytest.raises(TypeError, match=_SEED_TYPE_MESSAGE):
        call(dist, cast("int | None", seed))


def test_the_message_names_the_seed_it_rejected() -> None:
    """The offending value is in the message, so a computed seed is identifiable without a debugger."""
    with pytest.raises(ValueError, match=r"seed must be in \[0, 2\*\*63\), got -1$"):
        ps.Normal(0.0, 1.0).sample(seed=-1)


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.name)
@pytest.mark.parametrize("call", _CALLS.values(), ids=_CALLS.keys())
@pytest.mark.parametrize("path", ["scalar", "column"])
def test_the_largest_accepted_seed_reaches_every_sampler_plugin(
    spec: DistSpec, path: str, call: SamplingCallable
) -> None:
    """`_MAX_WIRE_INT` passes the guard *and* decodes on the wire, on all four plugin shapes."""
    dist = spec.make(spec.example) if path == "scalar" else spec.make_columns(spec.example)

    out = _FRAME.select(s=call(dist, _MAX_WIRE_INT))["s"]

    assert out.len() == _ROWS
    assert out.null_count() == 0


@pytest.mark.parametrize("call", _CALLS.values(), ids=_CALLS.keys())
def test_no_seed_is_still_accepted(call: SamplingCallable) -> None:
    """`seed=None` means OS entropy, not an out-of-range seed."""
    out = _FRAME.select(s=call(ps.Normal(0.0, 1.0), None))["s"]

    assert out.len() == _ROWS
    assert out.null_count() == 0


def test_the_bound_is_the_wire_limit_not_a_chosen_one() -> None:
    """One past `_MAX_WIRE_INT` really is undeliverable, so the guard is no stricter than it must be."""
    expr = ps.Normal(0.0, 1.0)._samples(size=1, seed=_MAX_WIRE_INT + 1)

    with pytest.raises(ComputeError, match="could not parse kwargs"):
        _FRAME.select(expr)


@pytest.mark.parametrize("seed", [None, 0])
@pytest.mark.parametrize("size", _NON_INTEGERS, ids=[type(size).__name__ for size in _NON_INTEGERS])
def test_a_non_integer_size_raises_at_call_time(size: object, seed: int | None) -> None:
    """`TypeError`, and before `_checked_seed`, so a bad `size` is reported even with a good seed."""
    with pytest.raises(TypeError, match=r"size should be an int, found "):
        ps.Normal(0.0, 1.0).samples(size=cast("int", size), seed=seed)


@pytest.mark.parametrize("size", [0, -1])
def test_a_non_positive_size_still_raises_value_error(size: int) -> None:
    """The pre-existing message is unchanged: eight per-distribution suites match on it."""
    with pytest.raises(ValueError, match="size must be a positive integer"):
        ps.Normal(0.0, 1.0).samples(size=size)


def test_size_is_checked_before_seed() -> None:
    """Argument order decides which of two bad arguments is reported, and it follows the signature."""
    with pytest.raises(TypeError, match="size should be an int"):
        ps.Normal(0.0, 1.0).samples(size=cast("int", 2.5), seed=-1)
