"""Constant-parameter fast paths stay byte-identical to the per-row path across polars contexts.

With constant parameters the whole expression is built from length-1 `pl.lit` inputs, so it is a scalar column and
polars decides per context what that means. `moment_test.py` and `value_keyed_test.py` pin the values under a plain
`select`; this module pins them where the *shape* differs, which a `select`-only test cannot see:

* **`over(group)`** broadcasts the scalar per partition, so both paths are full length there. The partitions here are
  deliberately uneven, so a path that assumed whole-frame length would diverge.
* **`group_by(group).agg(...)`** makes the scalar path one *scalar per group* (like `pl.col("x").mean()`) while the
  per-row path gives one *list per group*. That asymmetry is the contract; the list is constant, so its first element
  is the scalar.
* **the streaming engine** ingests the source in morsels rather than as one contiguous block.

A validating plugin that mishandled its length-1 input under partitioning, or a scalar path that stopped being a
scalar, would fail here while still passing the `select`-only suites.
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
from tests.property._specs import (
    ALL_SPECS,
    ULP_ABS_TOL,
    ULP_REL_TOL,
    ULP_TOLERANT_MOMENT_SPECS,
    ULP_TOLERANT_VALUE_SPECS,
)

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


def _assert_matches_across_contexts(
    frame: pl.DataFrame, fast: pl.Expr, slow: pl.Expr, *, fast_height: int, exact: bool = True
) -> None:
    """The scalar fast-path expr equals the per-row expr in every context the frame supports.

    `frame` must carry a `"g"` grouping column. Each context is checked independently so a failure
    names the context that diverged.

    `fast_height` is the fast path's own height under a plain `select`: 1 when every input is constant
    (a moment), the frame height when a column-valued `value` sets the length. Asserting it is what
    keeps this test pinning a *shape* and not only values. `exact` is the default because both paths
    compute the identical value; the caller relaxes it only for the `ULP_TOLERANT_*` specs.
    """
    # `select`: the fast path holds its own height, and broadcasts against the per-row column beside it.
    assert frame.select(r=fast).height == fast_height
    both = frame.select(fast=fast, slow=slow)
    assert both.height == frame.height
    assert_series_equal(
        both["fast"], both["slow"], check_names=False, check_exact=exact, rel_tol=ULP_REL_TOL, abs_tol=ULP_ABS_TOL
    )

    # `over`: polars broadcasts a scalar to each (uneven) partition's length, then scatters back.
    assert_series_equal(
        frame.select(r=fast.over("g"))["r"],
        frame.select(r=slow.over("g"))["r"],
        check_exact=exact,
        rel_tol=ULP_REL_TOL,
        abs_tol=ULP_ABS_TOL,
    )

    # `group_by().agg()`: a scalar expression aggregates to one scalar per group while the per-row path
    # gives one (constant) list per group; a full-length expression gives a list on both paths.
    grouped = frame.group_by("g", maintain_order=True).agg(fast=fast, slow=slow)
    if fast_height == 1:
        assert grouped.select(pl.col("slow").list.n_unique())["slow"].to_list() == [1] * grouped.height
        expected = grouped["slow"].list.first()
    else:
        expected = grouped["slow"]
    assert_series_equal(
        grouped["fast"], expected, check_names=False, check_exact=exact, rel_tol=ULP_REL_TOL, abs_tol=ULP_ABS_TOL
    )

    # streaming engine: the source is split across morsels; non-positional exprs must be invariant.
    if _STREAMING_AVAILABLE:
        lazy = frame.lazy().select(fast=fast, slow=slow).collect(engine="streaming")
        assert_series_equal(
            lazy["fast"], lazy["slow"], check_names=False, check_exact=exact, rel_tol=ULP_REL_TOL, abs_tol=ULP_ABS_TOL
        )


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
    # Every input is constant, so the moment is a scalar column.
    _assert_matches_across_contexts(frame, fast, slow, fast_height=1, exact=spec.name not in ULP_TOLERANT_MOMENT_SPECS)


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
        # The `value` argument is a column, so it sets the length on both paths.
        _assert_matches_across_contexts(
            frame, fast, slow, fast_height=_N_ROWS, exact=spec.name not in ULP_TOLERANT_VALUE_SPECS
        )
