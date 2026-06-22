"""Constant-parameter fast paths stay byte-identical to the per-row path across polars contexts.

The scalar moment and closed-form value-keyed fast paths (issue 20 item 6) build a length-1
validity gate, `pl.when(validated_once.is_not_null()).then(<context-length value>)`, where
`validated_once` is a validating plugin called on length-1 `pl.lit` inputs. The bit-equality tests
`moment_test.py` and `value_keyed_test.py` pin the fast path against the per-row path under a plain
`select`, where the gate broadcasts a length-1 condition against a whole-frame value.

This module pins the same equality under the contexts where the broadcast target length is *not* the
whole frame, the cases a `select`-only test cannot see:

* `over(group)` and `group_by(group).agg(...)` partition the frame, so `pl.len()` (the length the
  length-1 gate must broadcast to) differs per partition, and the partitions here are deliberately
  uneven;
* the streaming engine ingests the source in morsels rather than as one contiguous block.

A gate that silently assumed whole-frame length, or a validating plugin that mishandled its length-1
input under partitioning, would diverge from the per-row path here while still passing the
`select`-only suites.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from packaging.version import Version

from polars_stats.distributions._base import ContinuousDistribution, DiscreteDistribution
from tests._polars_compat import PL_VERSION, assert_series_equal, linear_space
from tests.property._specs import ALL_SPECS

if TYPE_CHECKING:
    from polars_stats.distributions._base import _UnivariateDistribution
    from tests.property._specs import DistSpec

_MOMENTS = ("mean", "variance", "std", "entropy")

# Uneven partitions on purpose: a length-1 validity gate that quietly broadcast against the whole
# frame instead of the partition would match on equal-sized groups and diverge here.
_GROUP_SIZES = (3, 1, 4, 2, 5)
_GROUPS = [g for g, size in enumerate(_GROUP_SIZES) for _ in range(size)]
_N_ROWS = len(_GROUPS)

# `engine="streaming"` is only selectable on recent polars; below this both code paths still run
# under the default (in-memory) engine via the eager contexts. Mirrors `sample_test.py`.
_STREAMING_AVAILABLE = Version("1.36.0") <= PL_VERSION


def _log_density(dist: _UnivariateDistribution, value: pl.Expr) -> pl.Expr:
    """`log_pdf` / `log_pmf` by family; the method lives on the family subclass, hence the narrowing."""
    if isinstance(dist, ContinuousDistribution):
        return dist.log_pdf(value)
    if isinstance(dist, DiscreteDistribution):
        return dist.log_pmf(value)
    msg = f"unsupported distribution family: {type(dist)}"  # pragma: no cover
    raise TypeError(msg)  # pragma: no cover


def _assert_matches_across_contexts(frame: pl.DataFrame, fast: pl.Expr, slow: pl.Expr) -> None:
    """The scalar fast-path expr equals the per-row expr in every context the frame supports.

    `frame` must carry a `"g"` grouping column. Each context is checked independently so a failure
    names the context that diverged. `check_exact` because the two paths compute the identical value
    (same Polars formula or same Rust body); only the validation differs, so any difference is a bug.
    """
    # `select`: the whole-frame context the other suites already cover, re-checked here as the anchor.
    assert_series_equal(frame.select(r=fast)["r"], frame.select(r=slow)["r"], check_exact=True)

    # `over`: the gate must broadcast to each (uneven) partition's length, then scatter back.
    assert_series_equal(frame.select(r=fast.over("g"))["r"], frame.select(r=slow.over("g"))["r"], check_exact=True)

    # `group_by().agg()`: the moment / value-keyed column aggregates to one list per group.
    grouped_fast = frame.group_by("g", maintain_order=True).agg(r=fast)["r"]
    grouped_slow = frame.group_by("g", maintain_order=True).agg(r=slow)["r"]
    assert_series_equal(grouped_fast, grouped_slow, check_exact=True)

    # streaming engine: the source is split across morsels; non-positional exprs must be invariant.
    if _STREAMING_AVAILABLE:
        lazy_fast = frame.lazy().select(r=fast).collect(engine="streaming")["r"]
        lazy_slow = frame.lazy().select(r=slow).collect(engine="streaming")["r"]
        assert_series_equal(lazy_fast, lazy_slow, check_exact=True)


@settings(max_examples=10)
@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.name)
@pytest.mark.parametrize("moment", _MOMENTS)
@given(data=st.data())
def test_moment_fast_path_matches_per_row_across_contexts(spec: DistSpec, moment: str, data: st.DataObject) -> None:
    """Each closed-form moment's scalar fast path equals the per-row path under every context."""
    params = data.draw(spec.params)
    frame = pl.DataFrame({"g": _GROUPS})

    fast = getattr(spec.make(params), moment)()
    slow = getattr(spec.make_columns(params), moment)()
    _assert_matches_across_contexts(frame, fast, slow)


@settings(max_examples=10)
@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.name)
@given(data=st.data())
def test_value_keyed_fast_path_matches_per_row_across_contexts(spec: DistSpec, data: st.DataObject) -> None:
    """Density / log-density / cdf / sf / ppf scalar fast paths equal the per-row path under every context.

    This is where the Uniform / Bernoulli closed forms (which route their *validation* through the
    same `_checked` gate as the moments) get their cross-context coverage: their value-keyed methods
    are pure Polars, so a broadcast bug in the scalar gate is the only way they could diverge.
    """
    params = data.draw(spec.params)
    scalar = spec.make(params)
    per_row = spec.make_columns(params)

    lo, hi = spec.eval_range(params)
    frame = pl.DataFrame(
        {
            "g": _GROUPS,
            "x": linear_space(lo, hi, _N_ROWS),
            "q": linear_space(0.05, 0.95, _N_ROWS),
        }
    )
    x, q = pl.col("x"), pl.col("q")
    cases = (
        (spec.density(scalar, x), spec.density(per_row, x)),
        (_log_density(scalar, x), _log_density(per_row, x)),
        (scalar.cdf(x), per_row.cdf(x)),
        (scalar.sf(x), per_row.sf(x)),
        (scalar.ppf(q), per_row.ppf(q)),
    )
    for fast, slow in cases:
        _assert_matches_across_contexts(frame, fast, slow)
